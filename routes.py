"""
routes.py – FastAPI router for the traffic analysis API.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
from typing import Any, AsyncGenerator, Dict, List

from fastapi import APIRouter, Response, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse

import db
import traffic_service
from annotator import PLACEHOLDER_FRAME

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """Returns the API health status."""
    return {"status": "ok", "pipeline_running": traffic_service.is_running()}


@router.get("/stats")
async def get_stats():
    """Returns the current traffic metrics snapshot."""
    state = traffic_service.get_state()
    
    # Calculate peak hour string (e.g., "2023-11-20 08")
    hourly = state["hourly_counts"]
    peak_hour = max(hourly, key=hourly.get) if hourly else "N/A"
    
    return {
        "metrics": state["metrics"],
        "currently_tracked": state["currently_tracked"],
        "speed_violations": state["speed_violations"],
        "wrong_way_alerts": state["wrong_way_alerts"],
        "peak_hour": peak_hour,
        "speed_distribution": state["speed_distribution"],
        "lane_counts": state["lane_counts"],
        "heatmap_enabled": state["heatmap_enabled"]
    }


@router.get("/vehicles")
async def get_recent_vehicles():
    """Return the 50 most recent crossing events."""
    return db.fetch_recent(50)


@router.get("/plates")
async def get_plates():
    """Return the most recent recognized license plates."""
    return traffic_service.get_plates()


@router.get("/alerts")
async def get_alerts():
    """Return the 20 most recent system alerts."""
    return traffic_service.get_alerts()


@router.post("/switch_camera")
async def switch_camera():
    """Switch the input source to the local webcam (0)."""
    logger.info("Switching source to webcam...")
    traffic_service.restart_pipeline("0")
    return {"status": "success", "source": "webcam"}


@router.get("/download-csv")
async def download_csv():
    """Export the entire traffic log as a CSV file."""
    rows = db.fetch_all()
    output = io.StringIO()
    output.write("id,timestamp,vehicle_type,direction,density\n")
    for r in rows:
        output.write(f"{r['id']},{r['timestamp']},{r['vehicle_type']},{r['direction']},{r['density']}\n")
    
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=traffic_data.csv"}
    )


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a new video file and restart the pipeline."""
    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    logger.info("Video uploaded: %s. Restarting pipeline...", file_path)
    # Restart the background pipeline with the new file
    traffic_service.restart_pipeline(file_path)
    
    return {"status": "success", "filename": file.filename}


@router.post("/toggle_heatmap")
async def toggle_heatmap():
    """Toggle the heatmap overlay state."""
    new_state = traffic_service.toggle_heatmap()
    return {"status": "success", "heatmap_enabled": new_state}


@router.get("/video_feed")
async def video_feed():
    """MJPEG streaming endpoint for the live video."""
    async def frame_stream() -> AsyncGenerator[bytes, None]:
        while True:
            frame_bytes = traffic_service.get_latest_frame()
            if frame_bytes is None:
                frame_bytes = PLACEHOLDER_FRAME
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            await asyncio.sleep(0.04)  # ~25 FPS

    return StreamingResponse(
        frame_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store"},
    )
