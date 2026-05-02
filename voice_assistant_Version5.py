#!/usr/bin/env python3
import os
import threading
import subprocess
import logging
import time
from typing import List, Tuple, Optional

import gradio as gr

# Configuration (adapt paths as needed)
PIPER_BIN = '/home/test/my-venv/venv-3.12/bin/piper'
PIPER_MODEL = '/home/test/.local/share/piper/models/en_US-lessac-high.onnx'
# Maximum characters per single piper invocation (avoid extremely long single calls).
PIPER_MAX_CHARS = 4000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Simple in-memory cache for uploaded files: key -> (mtime, size, lines)
_lines_cache = {}
_cache_lock = threading.Lock()

# Playback/process control
_stop_flag = False
_current_thread: Optional[threading.Thread] = None
_current_procs = {"piper": None, "paplay": None}
_procs_lock = threading.Lock()


def _cache_key_for_file(file_obj) -> Optional[Tuple[str, float, int]]:
    if file_obj is None:
        return None
    path = getattr(file_obj, "name", None)
    if not path or not os.path.exists(path):
        return None
    try:
        st = os.stat(path)
        return (path, st.st_mtime, st.st_size)
    except Exception:
        return None


def _get_lines_cached(file_obj) -> List[str]:
    """
    Return list of lines for file_obj, caching on disk mtime+size.
    """
    key = _cache_key_for_file(file_obj)
    if key is None:
        return []
    with _cache_lock:
        cached = _lines_cache.get(key[0])
        # cached is stored as (mtime, size, lines)
        if cached and cached[0] == key[1] and cached[1] == key[2]:
            return cached[2]
        # read and store
        with open(key[0], 'r', encoding='utf-8') as f:
            lines = f.readlines()
        _lines_cache[key[0]] = (key[1], key[2], lines)
        return lines


def get_chunks(file_path, chunk_size) -> Tuple[str, List[str]]:
    """
    chunk_size is interpreted as "lines per chunk" (int).
    Returns human-readable total and list of choice strings.
    """
    if file_path is None or chunk_size is None:
        return "Total lines: 0", []
    try:
        chunk_size = max(1, int(chunk_size))
    except Exception:
        chunk_size = 1

    lines = _get_lines_cached(file_path)
    total = len(lines)
    if total == 0:
        return "Total lines: 0", []

    n_chunks = (total + chunk_size - 1) // chunk_size
    choices = []
    for i in range(n_chunks):
        start = i * chunk_size + 1
        end = min((i + 1) * chunk_size, total)
        choices.append(f"Chunk {i+1}: Line {start} to {end}")
    return f"Total lines: {total}", choices


def show_chunk(file_path, chunk_choice) -> str:
    if file_path is None or not chunk_choice:
        return ""
    lines = _get_lines_cached(file_path)
    parts = chunk_choice.split(": Line ")
    if len(parts) < 2:
        return ""
    try:
        line_range = parts[1].strip()
        start_s, end_s = line_range.split(" to ")
        start, end = int(start_s), int(end_s)
    except Exception:
        return ""
    # guard bounds
    start = max(1, start)
    end = min(len(lines), end)
    if start > end:
        return ""
    return "".join(lines[start - 1:end])


def _terminate_processes():
    with _procs_lock:
        for key in list(_current_procs.keys()):
            proc = _current_procs.get(key)
            if proc:
                try:
                    proc.kill()
                except Exception:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                _current_procs[key] = None


def stop_tts():
    """
    External handler to stop playback immediately.
    """
    global _stop_flag
    logging.info("Stop requested.")
    _stop_flag = True
    _terminate_processes()
    # also join thread if present
    global _current_thread
    if _current_thread and _current_thread.is_alive():
        _current_thread.join(timeout=1.0)
        _current_thread = None


def _run_piper_to_paplay(text: str, rate: str = '22050'):
    """
    Run piper for `text` and pipe output to paplay. Blocks until playback finished
    or until _stop_flag is set (in which case it tries to kill child processes).
    """
    if not text:
        return

    piper_cmd = [PIPER_BIN, '--model', PIPER_MODEL, '--output-raw']
    paplay_cmd = ['paplay', '--rate', rate, '--format', 's16le', '--channels', '1', '--raw']

    with _procs_lock:
        try:
            piper_proc = subprocess.Popen(piper_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _current_procs['piper'] = piper_proc
        except FileNotFoundError as e:
            logging.exception("Piper binary not found: %s", e)
            return
        except Exception:
            logging.exception("Failed to start piper")
            _current_procs['piper'] = None
            return

        try:
            paplay_proc = subprocess.Popen(paplay_cmd, stdin=piper_proc.stdout)
            _current_procs['paplay'] = paplay_proc
        except FileNotFoundError as e:
            logging.exception("paplay not found: %s", e)
            try:
                piper_proc.kill()
            except Exception:
                pass
            _current_procs['paplay'] = None
            _current_procs['piper'] = None
            return
        except Exception:
            logging.exception("Failed to start paplay")
            try:
                piper_proc.kill()
            except Exception:
                pass
            _current_procs['paplay'] = None
            _current_procs['piper'] = None
            return

    # Write to piper stdin and close it so piper can start producing output
    try:
        piper_proc.stdin.write(text.encode('utf-8'))
        piper_proc.stdin.close()
    except Exception:
        logging.exception("Failed to write to piper stdin")
        _terminate_processes()
        return

    # Wait for playback to finish; but check stop_flag periodically
    try:
        while True:
            if _stop_flag:
                logging.info("Stop flag set; terminating processes")
                _terminate_processes()
                break
            # Poll paplay; if finished, break
            with _procs_lock:
                pap = _current_procs.get('paplay')
            if pap is None:
                break
            ret = pap.poll()
            if ret is not None:
                # paplay finished
                break
            time.sleep(0.1)
    finally:
        # cleanup
        _terminate_processes()


def _split_text_into_chunks(text: str, max_chars: int = PIPER_MAX_CHARS) -> List[str]:
    """
    Split text into conservative chunks (prefer sentence boundaries if possible).
    Fallback: simple fixed-size slices.
    """
    text = text.strip()
    if not text:
        return []
    chunks = []
    # try naive sentence-ish split by punctuation
    import re
    sentences = re.split(r'(?<=[\.\?\!]\s)', text)
    cur = ""
    for s in sentences:
        if len(cur) + len(s) <= max_chars:
            cur += s
        else:
            if cur:
                chunks.append(cur.strip())
            if len(s) > max_chars:
                # break long sentence into slices
                for i in range(0, len(s), max_chars):
                    chunks.append(s[i:i + max_chars].strip())
                cur = ""
            else:
                cur = s
    if cur:
        chunks.append(cur.strip())
    # final fallback
    if not chunks:
        for i in range(0, len(text), max_chars):
            chunks.append(text[i:i + max_chars])
    return [c for c in chunks if c]


def _tts_worker_for_chunk_text(text: str):
    """
    Background worker: split text into chunks and call piper->paplay per chunk.
    Allows early termination between chunks.
    """
    global _stop_flag
    try:
        chunks = _split_text_into_chunks(text, PIPER_MAX_CHARS)
        for i, chunk in enumerate(chunks):
            if _stop_flag:
                logging.info("Stopping before chunk %d", i)
                break
            logging.info("Playing chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))
            _run_piper_to_paplay(chunk)
            # small pause to allow responsive stop and avoid immediate re-spawning
            if _stop_flag:
                break
            time.sleep(0.05)
    except Exception:
        logging.exception("Error in TTS worker")
    finally:
        # ensure we reset the flag and procs on normal completion
        _terminate_processes()


def read_english_start(file_path, chunk_select, chunk_size) -> str:
    """
    Entrypoint called by Gradio when user clicks "Read English".
    This starts a background thread to play audio and immediately returns the chunk's text.
    """
    global _stop_flag, _current_thread
    _stop_flag = False

    if file_path is None or not chunk_select:
        return ""

    lines = _get_lines_cached(file_path)
    parts = chunk_select.split(": Line ")
    if len(parts) < 2:
        return ""

    try:
        line_range = parts[1].strip()
        start_s, end_s = line_range.split(" to ")
        start, end = int(start_s), int(end_s)
    except Exception:
        return ""

    start = max(1, start)
    end = min(len(lines), end)
    if start > end:
        return ""
    current_text = "".join(lines[start - 1:end])

    # Start background thread to play the text (non-blocking)
    logging.info("Starting TTS background thread for %d chars", len(current_text))
    worker = threading.Thread(target=_tts_worker_for_chunk_text, args=(current_text,), daemon=True)
    _current_thread = worker
    worker.start()

    return current_text


# Wiring Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# Voice Assistant App (patched)")
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload File")
            total_lines = gr.Textbox(label="", lines=1, interactive=False)
            chunk_size_input = gr.Number(label="Chunk Size (lines per chunk)", value=10, precision=0)
            chunk_select = gr.Radio(label="Select Chunk", choices=[])
        with gr.Column():
            text_output = gr.Textbox(label="File Content", lines=15)
    with gr.Row():
        read_button = gr.Button("Read English")
        stop_button = gr.Button("Stop", variant="stop")

    # Event handlers
    def on_input_change(file_path, chunk_size):
        total, choices = get_chunks(file_path, chunk_size)
        return total, gr.update(choices=choices, value=choices[0] if choices else "")

    file_input.change(fn=on_input_change, inputs=[file_input, chunk_size_input], outputs=[total_lines, chunk_select])
    chunk_size_input.change(fn=on_input_change, inputs=[file_input, chunk_size_input], outputs=[total_lines, chunk_select])
    chunk_select.change(fn=show_chunk, inputs=[file_input, chunk_select], outputs=text_output)

    # read_button starts playback in background and returns the text immediately
    read_button.click(fn=read_english_start, inputs=[file_input, chunk_select, chunk_size_input], outputs=text_output)
    stop_button.click(fn=stop_tts, inputs=[], outputs=[])

    # Use Gradio queue to allow concurrent requests without freezing the server
    app.queue()
    app.launch()