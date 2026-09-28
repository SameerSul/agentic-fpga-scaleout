"""Fetch a Qwen checkpoint's weights, config and tokenizer, on request only.

The repo runs without these. They are what lets qwen_real.py run the
actual model, not a checkpoint trained here, through the generated
blocks' arithmetic. Into qwen_weights/, which git ignores:

  qwen2.5   Qwen2.5-0.5B, about 1 GB, into qwen_weights/
  qwen3     Qwen3-0.6B, about 1.5 GB, into qwen_weights/qwen3-0.6b/: the
            model Architect Labs hosted on an FPGA

Run: python3 fetch_qwen.py [--model qwen2.5|qwen3]
"""
import argparse
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    "qwen2.5": (os.path.join(ROOT, "qwen_weights"),
                "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/"),
    "qwen3": (os.path.join(ROOT, "qwen_weights", "qwen3-0.6b"),
              "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/"),
}
DEST, BASE = MODELS["qwen2.5"]
FILES = ("config.json", "tokenizer.json", "model.safetensors")


def fetch(model="qwen2.5"):
    dest, base = MODELS[model]
    os.makedirs(dest, exist_ok=True)
    for name in FILES:
        path = os.path.join(dest, name)
        if os.path.exists(path):
            print("have", name)
            continue
        print("fetching", name)
        sys.stdout.flush()
        tmp = path + ".part"
        urllib.request.urlretrieve(base + name, tmp)
        os.replace(tmp, path)
        print("  %.1f MB" % (os.path.getsize(path) / 1e6))
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="qwen2.5")
    fetch(ap.parse_args().model)


if __name__ == "__main__":
    main()
