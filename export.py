#!/usr/bin/env python3
"""
Single entry point to export SAM3 to TensorRT -- dispatches on --prompt:

  --prompt EMPTY/omitted  -> export the OPEN-VOCAB architecture (dynamic
      prompt, classes NOT baked into the graph) via SAM3Open/export.py, into
      ../weight/sam3_{imgsz}_open/. Prompts are supplied at runtime via a
      text tensor (nvinfer keeps the same model file, only the input tensor
      content changes -- see SAM3Open/ds9_rtsp.py +
      SAM3Open/sam3open_text.py for the runtime pipeline that uses this),
      so this only needs to run once.

  --prompt <class...>     -> export the BAKED architecture (classes baked in
      as graph constants, the resulting engine runs directly in nvinfer just
      like YOLO) via SAM3Fixed/export.py. Changing the prompt requires
      re-exporting (or OTF-swapping to a different already-exported engine,
      see SAM3Fixed/spec_cache.py, used by SAM3Fixed/ds9_rtsp.py).

Both branches are subprocess calls into the existing scripts in each
subfolder -- no logic duplication, this is just a unified CLI.

Usage:
    python3 export.py --prompt truck car --mode det --imgsz 420
    python3 export.py --imgsz 1008              # no prompt -> open-vocab
"""
import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).parent
SAM3BAKED = _HERE / "SAM3Fixed"
SAM3OTF = _HERE / "SAM3Open"


def export_baked(args):
    cmd = [
        sys.executable, str(SAM3BAKED / "export.py"),
        "--classes", *args.prompt,
        "--mode", args.mode,
        "--imgsz", str(args.imgsz),
        "--opset", str(args.opset),
        "--device", args.device,
        "--min-batch", str(args.min_batch),
        "--opt-batch", str(args.opt_batch),
        "--max-batch", str(args.max_batch),
    ]
    if args.skip_trt:
        cmd.append("--skip-trt")
    print(f"[export] BAKED (prompt={args.prompt}) -> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=SAM3BAKED)


def export_open_vocab(args):
    cmd = [
        sys.executable, str(SAM3OTF / "export.py"),
        "--imgsz", str(args.imgsz),
        "--opset", str(args.opset),
        "--device", args.device,
        # no --output-dir: SAM3Open/export.py defaults to the shared
        # ../weight/sam3_{imgsz}_open/ folder
    ]
    print(f"[export] OPEN-VOCAB (no prompt) -> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=SAM3OTF)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", nargs="*", default=None,
                    help="class list (e.g. --prompt truck car). "
                         "Empty or omitted -> export open-vocab (dynamic-prompt).")
    p.add_argument("--mode", default="det", choices=["det", "seg"],
                    help="only applies to the --prompt (baked) branch; open-vocab is always det-only")
    p.add_argument("--imgsz", type=int, default=420)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--device", default="cuda")
    p.add_argument("--min-batch", type=int, default=1)
    p.add_argument("--opt-batch", type=int, default=4)
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--skip-trt", action="store_true", help="only applies to the --prompt (baked) branch")
    args = p.parse_args()

    if args.prompt:
        export_baked(args)
    else:
        export_open_vocab(args)


if __name__ == "__main__":
    main()
