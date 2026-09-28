"""Fetch Qwen2.5-0.5B's weights, config and tokenizer, on request only.

The repo runs without these. They are what lets qwen_real.py run the
actual model, not a checkpoint trained here, through the generated
blocks' arithmetic. About 1 GB, into qwen_weights/, which git ignores.

Run: python3 fetch_qwen.py
"""
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(ROOT, "qwen_weights")
BASE = "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/"
FILES = ("config.json", "tokenizer.json", "model.safetensors")


def main():
    os.makedirs(DEST, exist_ok=True)
    for name in FILES:
        path = os.path.join(DEST, name)
        if os.path.exists(path):
            print("have", name)
            continue
        print("fetching", name)
        sys.stdout.flush()
        tmp = path + ".part"
        urllib.request.urlretrieve(BASE + name, tmp)
        os.replace(tmp, path)
        print("  %.1f MB" % (os.path.getsize(path) / 1e6))


if __name__ == "__main__":
    main()
