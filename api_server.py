"""
api_server.py — FastAPI backend for the Surveillance AI Web Dashboard.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

ROOT       = Path(__file__).parent
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "output"
WEB_DIR    = ROOT / "web"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="ViViT Surveillance AI Dashboard", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Job state
# ─────────────────────────────────────────────────────────────

class JobState:
    def __init__(self, job_id: str, source_path: str):
        self.job_id       = job_id
        self.source_path  = source_path
        self.status       = "queued"
        self.events: List[dict] = []
        self.results: Optional[dict] = None
        self.output_video: Optional[str] = None
        self.events_video: Optional[str] = None
        self._lock = threading.Lock()

    def push(self, event: dict):
        with self._lock:
            self.events.append(event)

    def pop_new(self, cursor: int) -> List[dict]:
        with self._lock:
            return list(self.events[cursor:])


JOBS: Dict[str, JobState] = {}

# ─────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────

_PIPELINE_INSTANCE = None
_PIPELINE_LOCK = threading.Lock()

def _get_pipeline(enable_llm, llm_model, llm_4bit, hf_token, llm_use_api=True):
    global _PIPELINE_INSTANCE
    if _PIPELINE_INSTANCE is None:
        from pipeline import HierarchicalPipeline
        from config import CFG
        _PIPELINE_INSTANCE = HierarchicalPipeline(
            cfg        = CFG,
            enable_llm = enable_llm,
            llm_model  = llm_model,
            llm_4bit   = llm_4bit,
            hf_token   = hf_token or os.environ.get("HF_TOKEN"),
            llm_use_api= llm_use_api,
        )
    return _PIPELINE_INSTANCE


def _run_pipeline(job: JobState, enable_llm, llm_model, llm_4bit, hf_token, llm_use_api=True):
    """Runs in background thread. Pushes SSE events into job.events."""
    import builtins
    _orig_print = builtins.print

    def _captured_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        _orig_print(msg)
        job.push({"type": "log", "message": msg})

    builtins.print = _captured_print
    try:
        job.status = "running"
        job.push({"type": "status", "status": "running", "message": "Loading models..."})

        with _PIPELINE_LOCK:
            pipeline = _get_pipeline(enable_llm, llm_model, llm_4bit, hf_token)

        stem     = Path(job.source_path).stem
        out_path = str(OUTPUT_DIR / f"{stem}_{job.job_id[:8]}_output.mp4")
        ev_path  = str(OUTPUT_DIR / f"{stem}_{job.job_id[:8]}_events.mp4")

        job.push({"type": "status", "status": "processing", "message": "Pipeline running..."})

        results = pipeline.run(
            source       = job.source_path,
            output_path  = out_path,
            show_preview = False,
            max_frames   = 0,
        )

        track_data = []
        fps = pipeline.cfg.tracker.frame_rate or 30.0
        for tid, res in sorted(results.items()):
            fs, fe = pipeline._track_frame_ranges.get(tid, (0, 0))
            t_start = float(fs) / fps
            t_end = float(fe) / fps
            t = {
                "track_id":     tid,
                "binary_label": res.binary_label,
                "binary_conf":  round(res.binary_conf, 3),
                "fine_label":   res.fine_label,
                "fine_conf":    round(res.fine_conf, 3),
                "is_abnormal":  res.is_abnormal,
                "routed_to":    getattr(res, "routed_to", ""),
                "frame_start":  fs,
                "frame_end":    fe,
                "t_start":      round(t_start, 2),
                "t_end":        round(t_end, 2),
            }
            track_data.append(t)
            job.push({
                "type":        "track",
                "track_id":    tid,
                "label":       res.fine_label,
                "conf":        round(res.fine_conf, 3),
                "binary":      res.binary_label,
                "is_abnormal": res.is_abnormal,
                "routed_to":   getattr(res, "routed_to", ""),
                "frame_start":  fs,
                "frame_end":    fe,
                "t_start":      round(t_start, 2),
                "t_end":        round(t_end, 2),
            })

        if os.path.exists(out_path):
            job.output_video = out_path
        if os.path.exists(ev_path):
            job.events_video = ev_path

        # Read LLM report
        report_txt = ""
        report_dir = OUTPUT_DIR / "reports" / stem
        if report_dir.exists():
            all_txt = report_dir / "all_reports.txt"
            if all_txt.exists():
                report_txt = all_txt.read_text(encoding="utf-8")

        n_ab = sum(1 for t in track_data if t["is_abnormal"])
        job.results = {
            "job_id":       job.job_id,
            "source":       job.source_path,
            "total_tracks": len(track_data),
            "abnormal":     n_ab,
            "tracks":       track_data,
            "llm_report":   report_txt,
            "output_video": f"/api/video/{job.job_id}"        if job.output_video else None,
            "events_video": f"/api/events-video/{job.job_id}" if job.events_video else None,
        }
        job.push({
            "type":         "complete",
            "total_tracks": len(track_data),
            "abnormal":     n_ab,
            "llm_report":   report_txt,
            "output_video": job.results["output_video"],
            "events_video": job.results["events_video"],
        })
        job.status = "done"

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        job.push({"type": "error", "message": str(exc), "traceback": tb})
        job.status = "error"
        _orig_print(f"[API] Job {job.job_id} error: {exc}\n{tb}")
    finally:
        builtins.print = _orig_print


# ─────────────────────────────────────────────────────────────
# API routes  (MUST come before static mount)
# ─────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    allowed_ext = {".mp4", ".avi", ".mov", ".mkv"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_ext:
        raise HTTPException(400, f"Unsupported format '{ext}'. Use: {allowed_ext}")

    job_id    = str(uuid.uuid4())
    save_path = str(UPLOAD_DIR / f"{job_id}{ext}")

    with open(save_path, "wb") as fh:
        while chunk := await file.read(1024 * 1024):
            fh.write(chunk)

    job = JobState(job_id=job_id, source_path=save_path)
    JOBS[job_id] = job

    cfg = _API_CFG
    threading.Thread(
        target  = _run_pipeline,
        args    = (job, cfg["enable_llm"], cfg["llm_model"], cfg["llm_4bit"], cfg["hf_token"]),
        daemon  = True,
        name    = f"pipeline-{job_id[:8]}",
    ).start()

    return JSONResponse({"job_id": job_id, "filename": file.filename})

def _ensure_job_loaded(job_id: str) -> None:
    if job_id in JOBS:
        return

    # Check if there is a summary.json and output files on disk
    video_path = None
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        p = UPLOAD_DIR / f"{job_id}{ext}"
        if p.exists():
            video_path = str(p)
            break

    if not video_path:
        out_candidates = list(OUTPUT_DIR.glob(f"*_{job_id[:8]}_output.mp4"))
        if out_candidates:
            video_path = str(UPLOAD_DIR / f"{job_id}.mp4")
        else:
            return

    report_dir = OUTPUT_DIR / "reports" / job_id
    summary_path = report_dir / "summary.json"
    all_reports_path = report_dir / "all_reports.txt"

    if not summary_path.exists():
        return

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        
        report_txt = ""
        if all_reports_path.exists():
            report_txt = all_reports_path.read_text(encoding="utf-8")

        out_video = None
        ev_video = None
        out_candidates = list(OUTPUT_DIR.glob(f"*_{job_id[:8]}_output.mp4"))
        if out_candidates:
            out_video = str(out_candidates[0])
        ev_candidates = list(OUTPUT_DIR.glob(f"*_{job_id[:8]}_events.mp4"))
        if ev_candidates:
            ev_video = str(ev_candidates[0])

        job = JobState(job_id=job_id, source_path=video_path)
        job.status = "done"
        job.output_video = out_video
        job.events_video = ev_video

        track_data = []
        for t in summary.get("tracks", []):
            td = {
                "track_id":     t.get("track_id"),
                "binary_label": t.get("binary_label", "Abnormal" if t.get("is_abnormal") else "Normal"),
                "binary_conf":  round(t.get("binary_conf", 1.0), 3),
                "fine_label":   t.get("fine_label"),
                "fine_conf":    round(t.get("fine_conf", 1.0), 3),
                "is_abnormal":  t.get("is_abnormal", False),
                "routed_to":    t.get("routed_to", "finegrained" if t.get("fine_label") else "binary"),
                "frame_start":  t.get("frame_start", 0),
                "frame_end":    t.get("frame_end", 0),
                "t_start":      round(t.get("t_start", 0.0), 2),
                "t_end":        round(t.get("t_end", 0.0), 2),
            }
            track_data.append(td)
            
        job.results = {
            "job_id":       job_id,
            "source":       video_path,
            "total_tracks": len(track_data),
            "abnormal":     summary.get("abnormal_tracks", 0),
            "tracks":       track_data,
            "llm_report":   report_txt,
            "output_video": f"/api/video/{job_id}"        if out_video else None,
            "events_video": f"/api/events-video/{job_id}" if ev_video else None,
        }
        
        job.push({
            "type":         "complete",
            "total_tracks": len(track_data),
            "abnormal":     summary.get("abnormal_tracks", 0),
            "llm_report":   report_txt,
            "output_video": job.results["output_video"],
            "events_video": job.results["events_video"],
        })
        
        JOBS[job_id] = job
    except Exception as e:
        print(f"Error loading job {job_id} from disk: {e}")


@app.get("/api/stream/{job_id}")
async def stream_progress(job_id: str):
    _ensure_job_loaded(job_id)
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    job = JOBS[job_id]

    async def gen():
        cursor = 0
        while True:
            events = job.pop_new(cursor)
            for ev in events:
                yield f"data: {json.dumps(ev)}\n\n"
                cursor += 1
                if ev.get("type") in ("complete", "error"):
                    return
            if job.status in ("done", "error") and cursor >= len(job.events):
                return
            await asyncio.sleep(0.25)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    _ensure_job_loaded(job_id)
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    job = JOBS[job_id]
    return JSONResponse({"status": job.status, "results": job.results})


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    _ensure_job_loaded(job_id)
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    return {"job_id": job_id, "status": JOBS[job_id].status}


@app.get("/api/video/{job_id}")
async def get_video(job_id: str):
    _ensure_job_loaded(job_id)
    if job_id not in JOBS or not JOBS[job_id].output_video:
        raise HTTPException(404, "Video not ready")
    return FileResponse(JOBS[job_id].output_video, media_type="video/mp4")


@app.get("/api/events-video/{job_id}")
async def get_events_video(job_id: str):
    _ensure_job_loaded(job_id)
    if job_id not in JOBS or not JOBS[job_id].events_video:
        raise HTTPException(404, "Events video not ready")
    return FileResponse(JOBS[job_id].events_video, media_type="video/mp4")


@app.get("/api/source-video/{job_id}")
async def get_source_video(job_id: str):
    _ensure_job_loaded(job_id)
    if job_id not in JOBS or not JOBS[job_id].source_path:
        raise HTTPException(404, "Source video not found")
    return FileResponse(JOBS[job_id].source_path, media_type="video/mp4")


@app.get("/api/health")
async def health():
    return {"status": "ok", "jobs": len(JOBS)}


# ─────────────────────────────────────────────────────────────
# Static files  (mount LAST — catches everything else)
# ─────────────────────────────────────────────────────────────

if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="static")
else:
    @app.get("/")
    async def root():
        return HTMLResponse("<h2>web/ directory not found. Check d:/idk/web/</h2>")


# ─────────────────────────────────────────────────────────────
# Config store (filled by __main__ before server starts)
# ─────────────────────────────────────────────────────────────

_API_CFG: dict = {
    "enable_llm": False,
    "llm_model":  "meta-llama/Llama-3.1-8B-Instruct",
    "llm_4bit":   False,
    "hf_token":   os.environ.get("HF_TOKEN"),   # Set via environment variable: export HF_TOKEN=hf_...
}

# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ViViT Surveillance AI Server")
    p.add_argument("--host",      default="0.0.0.0")
    p.add_argument("--port",      type=int, default=8000)
    p.add_argument("--llm",       action="store_true")
    p.add_argument("--llm-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--llm-4bit",  action="store_true")
    p.add_argument("--hf-token",  default=None,
                   help="HuggingFace token (or set HF_TOKEN env variable)")
    args = p.parse_args()

    _API_CFG.update({
        "enable_llm": args.llm,
        "llm_model":  args.llm_model,
        "llm_4bit":   args.llm_4bit,
        "hf_token":   args.hf_token or os.environ.get("HF_TOKEN") or _API_CFG["hf_token"],
    })

    print(f"\n{'='*56}")
    print(f"  ViViT Surveillance AI Dashboard")
    print(f"  >> http://localhost:{args.port}")
    print(f"  LLM : {'ON  (' + args.llm_model + ')' if args.llm else 'OFF'}")
    print(f"  Web : {WEB_DIR}  ({'exists' if WEB_DIR.exists() else 'MISSING'})")
    print(f"{'='*56}\n")

    # uvicorn.Server.run() is synchronous and handles Windows signals correctly
    config = uvicorn.Config(
        app,
        host      = args.host,
        port      = args.port,
        log_level = "info",
    )
    server = uvicorn.Server(config)
    server.run()
