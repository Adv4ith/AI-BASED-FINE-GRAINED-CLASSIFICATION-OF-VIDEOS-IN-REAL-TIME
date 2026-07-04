"""
run_final.py — Production entry-point for the ViViT surveillance pipeline.

Pipeline stages:
  Stage 1  -> YOLOv8  person detection
  Stage 2  -> ByteTrack  multi-person tracking
  Stage 3  -> TubeManager  32-frame action tube buffering
  Stage 4a -> Binary ViViT  (Normal / Abnormal gate)
  Stage 4b -> Fine-grained ViViT  (5-class classification)
  Stage 5  -> Llama 3.1 LLM  surveillance report (optional)

Usage
-----
  python run_final.py --source video.mp4
  python run_final.py --source video.mp4 --llm
  python run_final.py --source video.mp4 --preview
  python run_final.py --source 0 --preview --no-save
"""

import argparse
import os
import sys
import time

from pipeline import HierarchicalPipeline
from config   import CFG


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ViViT Surveillance Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input / Output
    p.add_argument("--source", "-s", required=True,
                   help="Video file path or webcam index (e.g. 0)")
    p.add_argument("--output", "-o", default=None,
                   help="Output video path (default: output/<stem>_output.mp4)")
    p.add_argument("--preview", "-p", action="store_true",
                   help="Show OpenCV preview window while processing")
    p.add_argument("--no-save", action="store_true",
                   help="Do not save output video")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after N frames (0 = full video)")

    # Device
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "cpu"])

    # Detection / Tracking
    p.add_argument("--yolo-model", default=CFG.detector.model_name)
    p.add_argument("--yolo-conf", type=float, default=CFG.detector.conf_threshold)

    # ViViT Inference
    p.add_argument("--binary-thresh", type=float,
                   default=CFG.inference.binary_confidence_threshold)
    p.add_argument("--inference-stride", type=int,
                   default=CFG.tube.inference_stride)

    # LLM
    p.add_argument("--llm", action="store_true",
                   help="Enable Llama 3.1 surveillance reports")
    p.add_argument("--llm-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--llm-4bit", action="store_true",
                   help="4-bit quantization (needs bitsandbytes)")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace token (or set HF_TOKEN env variable)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    CFG.inference.device                      = args.device
    CFG.inference.binary_confidence_threshold = args.binary_thresh
    CFG.detector.model_name                   = args.yolo_model
    CFG.detector.conf_threshold               = args.yolo_conf
    CFG.tube.inference_stride                 = args.inference_stride
    if args.no_save:
        CFG.output.save_video = False

    try:
        source = int(args.source)
    except ValueError:
        source = args.source
        if not os.path.isfile(source):
            print(f"[ERROR] File not found: {source}", file=sys.stderr)
            sys.exit(1)

    print("\n" + "=" * 60)
    print("  ViViT Surveillance Pipeline")
    print("=" * 60)
    print(f"  Source      : {source}")
    print(f"  Device      : {args.device}")
    print(f"  LLM Reports : {'ON  (' + args.llm_model + ')' if args.llm else 'OFF'}")
    print("=" * 60 + "\n")

    pipeline = HierarchicalPipeline(
        cfg        = CFG,
        enable_llm = args.llm,
        llm_model  = args.llm_model,
        llm_4bit   = args.llm_4bit,
        hf_token   = args.hf_token or os.environ.get("HF_TOKEN"),
    )

    t0 = time.perf_counter()
    results = pipeline.run(
        source       = source,
        output_path  = args.output,
        show_preview = args.preview,
        max_frames   = args.max_frames,
    )
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print(f"  COMPLETE  |  {len(results)} tracks  |  {elapsed:.1f}s")
    print("=" * 60)

    normal_tracks   = [r for r in results.values() if not r.is_abnormal]
    abnormal_tracks = [r for r in results.values() if r.is_abnormal]
    print(f"\n  Normal tracks   : {len(normal_tracks)}")
    print(f"  Abnormal tracks : {len(abnormal_tracks)}")

    if abnormal_tracks:
        print("\n  ABNORMAL EVENTS DETECTED:")
        for r in sorted(abnormal_tracks, key=lambda x: x.fine_conf, reverse=True):
            print(f"    Track {r.track_id:3d}  >>  {r.fine_label:20s}  conf={r.fine_conf:.0%}")

    stem = (os.path.splitext(os.path.basename(str(source)))[0]
            if isinstance(source, str) else f"webcam_{source}")
    out_dir = CFG.output.output_dir
    print("\n  Output files:")
    if CFG.output.save_video:
        print(f"    Annotated video >> {out_dir}\\{stem}_output.mp4")
    if args.llm:
        print(f"    LLM reports     >> {out_dir}\\reports\\{stem}\\")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
