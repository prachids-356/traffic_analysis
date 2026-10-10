# Deployment and Setup Guide

This document explains how to run the project locally, how to deploy it in a containerized environment, and what to check if the dashboard does not appear or the service does not start.

## Local startup

### 1. Open a terminal in the project root

```powershell
cd "C:\Users\p5726\Downloads\traffic_analysis-main (1)\traffic_analysis-main"
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Run the app

```powershell
python main.py
```

### 4. Open the dashboard

```text
http://localhost:8000/
```

---

## Health checks

After startup, verify that the backend is alive:

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8000/healthz
```

Expected response:

```json
{"status":"ok"}
```

---

## Environment notes

### Python version
The project was validated successfully using Python 3.14 in this workspace.

### Required packages
The root requirements file includes:

- `fastapi`
- `uvicorn[standard]`
- `opencv-python-headless`
- `ultralytics`
- `easyocr`
- `pandas`
- `lap`
- `python-multipart`

If startup fails with import errors, reinstall the requirements file again before making changes to the code.

---

## Docker setup

This repository includes a Dockerfile for running the project in a containerized environment.

### Build the image

```powershell
docker build -t traffic-vision-ai .
```

### Run the image

```powershell
docker run -p 8000:8000 traffic-vision-ai
```

Then open:

```text
http://localhost:8000/
```

---

## Common troubleshooting

### Issue: `ModuleNotFoundError: No module named 'cv2'`

Fix:

```powershell
python -m pip install -r requirements.txt
```

### Issue: `No such file or directory` when starting Python

Fix:

```powershell
cd "<project-folder>"
python main.py
```

### Issue: App starts but dashboard does not load

Check:

- the app is still running
- the backend is listening on port 8000
- the browser is accessing http://localhost:8000/
- the model file exists in the root folder as `yolov8n.pt`

### Issue: Detection pipeline is slow

Possible fixes:

- lower the camera resolution
- reduce inference frequency
- skip OCR if not required
- use smaller videos for local testing

---

## Suggested production improvements

The current version works well for a local demo and internal validation. For a production deployment, the following improvements would be useful:

- use a managed datastore instead of the local SQLite database
- add authentication and admin controls
- add proper logging and metrics collection
- separate pipeline workers from the web API service
- add configuration profiles for different cameras or environments
- create automated test coverage for route and detection logic

---

## Final check

A successful startup is confirmed when:

1. the Python process is running,
2. the FastAPI app responds on port 8000,
3. the dashboard loads in the browser,
4. the health endpoint returns `200 OK`.
