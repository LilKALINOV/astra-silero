"""Silero TTS inference worker — JSON-lines over stdio.

The plugin's Rust process starts this script once and keeps it alive. Each
request arrives as one JSON line on stdin; each response leaves as one JSON
line on stdout. The model is a Silero *torch package* (`.pt`), so this needs
only `torch` — no pip `silero`, no torchaudio, no omegaconf. The first
synthesis downloads the chosen model into `.models/` next to the plugin
(e.g. v5_5_ru.pt, ~140 MB); after that everything is offline.

Request : {"cmd":"synthesize","text":"...","speaker":"xenia",
            "model":"v5_5_ru","sample_rate":24000,"out":"C:\\...\\tmp.wav"}
Response: {"ok":true,"duration_ms":1234}
          {"ok":false,"error":"..."}

On startup it prints {"ready":true} after torch imports.
"""

import json
import os
import re
import shutil
import struct
import sys
import threading
import urllib.request
import wave

import torch

BASE_URL = "https://models.silero.ai/models/tts/ru"
MODELS = {"v5_5_ru": "v5_5_ru.pt", "v4_ru": "v4_ru.pt"}
DEFAULT_MODEL = "v5_5_ru"
SPEAKERS = {"aidar", "baya", "kseniya", "xenia", "eugene"}
SAMPLE_RATES = {8000, 24000, 48000}
DEFAULT_SAMPLE_RATE = 24000

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE = os.path.abspath(
    os.environ.get("SILERO_CACHE_DIR") or os.path.join(_HERE, "..", ".models")
)
_loaded = {}
_download_lock = threading.Lock()


def log(msg):
    sys.stderr.write("[silero] %s\n" % msg)
    sys.stderr.flush()


def model_path(model_id):
    return os.path.join(_CACHE, MODELS[model_id])


def _download(url, dest):
    tmp = dest + ".part"
    log("downloading %s" % url)
    request = urllib.request.Request(url, headers={"User-Agent": "silero-tts-astra/0.1"})
    with urllib.request.urlopen(request, timeout=600) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    os.replace(tmp, dest)


def get_model(model_id):
    if model_id not in _loaded:
        os.makedirs(_CACHE, exist_ok=True)
        path = model_path(model_id)
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            # The prefetch thread may be downloading the same file; re-check
            # under the lock so `.part` is never written by two threads at once.
            with _download_lock:
                if not os.path.exists(path) or os.path.getsize(path) < 1000:
                    _download(BASE_URL + "/" + MODELS[model_id], path)
        package = torch.package.PackageImporter(path)
        _loaded[model_id] = package.load_pickle("tts_models", "model")
    return _loaded[model_id]


def _prefetch_default_model():
    try:
        os.makedirs(_CACHE, exist_ok=True)
        path = model_path(DEFAULT_MODEL)
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            with _download_lock:
                if not os.path.exists(path) or os.path.getsize(path) < 1000:
                    _download(BASE_URL + "/" + MODELS[DEFAULT_MODEL], path)
    except Exception as exc:
        log("prefetch failed: %r" % (exc,))


_TRANSLIT = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г",
    "h": "х", "i": "и", "j": "й", "k": "к", "l": "л", "m": "м", "n": "н",
    "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т", "u": "у",
    "v": "в", "w": "в", "x": "х", "y": "ы", "z": "з",
}
_TRANSLIT.update({k.upper(): v.upper() for k, v in list(_TRANSLIT.items())})
_TRANSLIT_TABLE = str.maketrans(_TRANSLIT)


def _to_cyrillic(text):
    """Map Latin letters onto Cyrillic so the RU-only model keeps any text."""
    s = text.translate(_TRANSLIT_TABLE)
    if not re.search(r"[а-яёА-ЯЁ]", s):
        raise ValueError("empty text: no readable characters after normalization")
    return s


def synthesize(text, speaker, model_id, sample_rate, out):
    model = get_model(model_id)
    try:
        audio = model.apply_tts(text=_to_cyrillic(text), speaker=speaker, sample_rate=sample_rate)
    except ValueError as exc:
        raise ValueError("the model refused the text: %s" % (exc or "no acceptable characters")) from exc
    # int16 little-endian samples without numpy: pack the flat list.
    samples = (
        audio.cpu()
        .detach()
        .mul(32767.0)
        .clamp(-32768.0, 32767.0)
        .to(torch.int16)
        .flatten()
    )
    data = struct.pack("<%dh" % samples.numel(), *samples.tolist())
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(data)
    return round(len(data) / (2 * sample_rate) * 1000)


def respond(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def handle(req):
    cmd = req.get("cmd")
    if cmd != "synthesize":
        return {"ok": False, "error": "unknown command %r" % (cmd,)}
    text = req.get("text")
    speaker = req.get("speaker") or "xenia"
    model_id = req.get("model") or DEFAULT_MODEL
    sample_rate = int(req.get("sample_rate") or DEFAULT_SAMPLE_RATE)
    out = req.get("out")
    out_dir = os.path.dirname(out)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty text")
    if speaker not in SPEAKERS:
        raise ValueError("unknown speaker %r" % (speaker,))
    if model_id not in MODELS:
        raise ValueError("unknown model %r (have: %s)" % (model_id, ", ".join(sorted(MODELS))))
    if sample_rate not in SAMPLE_RATES:
        raise ValueError("unsupported sample_rate %r" % (sample_rate,))
    if not out or not os.path.isdir(out_dir):
        raise ValueError("bad out path %r" % (out,))
    duration_ms = synthesize(text, speaker, model_id, sample_rate, out)
    return {"ok": True, "duration_ms": duration_ms}


def main():
    log(
        "worker up: python %s, torch %s, cache %s"
        % (sys.version.split()[0], torch.__version__, _CACHE)
    )
    respond({"ready": True})
    threading.Thread(target=_prefetch_default_model, daemon=True).start()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            respond({"ok": False, "error": "bad json: %r" % (exc,)})
            continue
        try:
            respond(handle(req))
        except Exception as exc:
            log("error: %r" % (exc,))
            respond({"ok": False, "error": str(exc)})


if __name__ == "__main__":
    main()