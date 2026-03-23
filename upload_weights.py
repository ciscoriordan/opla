#!/usr/bin/env python3
"""Upload Opla weights to HuggingFace.

Usage:
    python upload_weights.py                # upload grc weights
    python upload_weights.py --lang med     # upload med weights
    python upload_weights.py --lang all     # upload all available
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "ciscoriordan/opla"
WEIGHTS_DIR = Path(__file__).parent / "weights"


def upload_lang(api, lang):
    """Upload weights for a single language (PyTorch + ONNX if available)."""
    weight_file = WEIGHTS_DIR / lang / f"opla_{lang}.pt"
    if not weight_file.exists():
        print(f"  Skipping {lang}: {weight_file} not found")
        return False

    # Upload PyTorch weights
    remote_path = f"weights/{lang}/opla_{lang}.pt"
    print(f"  Uploading {weight_file} -> {remote_path} ({weight_file.stat().st_size / 1e6:.0f} MB)")
    api.upload_file(
        path_or_fileobj=str(weight_file),
        path_in_repo=remote_path,
        repo_id=REPO_ID,
    )
    print(f"  Done: {lang} (PyTorch)")

    # Upload ONNX files if available
    onnx_dir = WEIGHTS_DIR / lang / "onnx"
    if onnx_dir.exists():
        onnx_files = list(onnx_dir.iterdir())
        if onnx_files:
            print(f"  Uploading ONNX files from {onnx_dir}:")
            for f in sorted(onnx_files):
                remote = f"weights/{lang}/onnx/{f.name}"
                print(f"    {f.name} -> {remote} ({f.stat().st_size / 1e6:.1f} MB)")
                api.upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=remote,
                    repo_id=REPO_ID,
                )
            print(f"  Done: {lang} (ONNX)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Upload Opla weights to HuggingFace")
    parser.add_argument("--lang", default="grc", choices=["grc", "med", "all"],
                        help="Which weights to upload")
    parser.add_argument("--create-repo", action="store_true",
                        help="Create the HF repo if it doesn't exist")
    args = parser.parse_args()

    api = HfApi()

    if args.create_repo:
        api.create_repo(REPO_ID, exist_ok=True)
        print(f"Repo: {REPO_ID}")

    langs = ["grc", "med"] if args.lang == "all" else [args.lang]

    for lang in langs:
        print(f"\n{lang}:")
        upload_lang(api, lang)

    print(f"\nWeights available at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
