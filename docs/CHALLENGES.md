# Challenges Faced and How They Were Resolved

This document captures the main difficulties encountered while building and running the traffic analytics project, along with the practical fixes that made the system work reliably.

## 1. App could not start because the project was launched from the wrong directory

### Problem
The app was first attempted from the parent folder instead of the project root. The terminal command used:

```powershell
python main.py
```

but the file did not exist at that location. This caused a `No such file or directory` error.

### Resolution
The project was launched from the actual workspace folder:

```powershell
cd "C:\Users\p5726\Downloads\traffic_analysis-main (1)\traffic_analysis-main"
python main.py
```

Once the correct working directory was used, the service started normally.

---

## 2. Missing OpenCV dependency prevented the app from importing

### Problem
The first real runtime error was:

```text
ModuleNotFoundError: No module named 'cv2'
```

The app imports `cv2` from OpenCV and failed immediately at startup.

### Resolution
The project dependencies were installed from the repository requirements file:

```powershell
python -m pip install -r requirements.txt
```

This included the required OpenCV package:

- `opencv-python-headless`
- `ultralytics`
- `easyocr`
- `lap`
- other runtime dependencies needed by the detection stack

After installation, the application was able to import successfully.

---

## 3. Dependency installation was needed even though the environment looked partially prepared

### Problem
The environment showed some packages already installed, but the project still failed at import time because the specific libraries required for detection were missing or incomplete. This is a common issue when a Python environment is not created from the repository's declared dependency list.

### Resolution
Rather than debugging module by module, the clean fix was to install the exact project requirements. This ensured compatibility among:

- FastAPI
- Uvicorn
- OpenCV
- YOLO
- EasyOCR
- pandas
- NumPy
- tracking libraries

This created a reproducible environment and reduced startup issues significantly.

---

## 4. The runtime had to be verified by checking the health endpoint instead of assuming the app was ready

### Problem
After the install completed, the app still needed confirmation that the service was actually listening and responding, rather than just that the Python process had launched.

### Resolution
The health endpoint was checked directly:

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8000/healthz
```

This returned:

```json
{"status":"ok"}
```

With HTTP 200, it was confirmed that the app was correctly serving requests from the dashboard backend.

---

## 5. The project includes multiple app layouts, which created confusion about the real entry point

### Problem
The workspace contained both a root-level app and a nested `traffic_ai/` package. This made it unclear which module was the canonical running service and which one was a reference or prototype structure.

### Resolution
The project was treated as the root application when running the dashboard:

- main app entry: `main.py`
- runtime route source: `routes.py`
- background worker: `traffic_service.py`

The nested `traffic_ai/` folder was documented as a modular packaging variant rather than the active runtime entry point for the current deployment.

---

## 6. Heavy ML workloads required a realistic expectation about compute time

### Problem
The project uses YOLO-based detection and object tracking, which can be resource-heavy on a local machine, especially when running live traffic inference continuously. This can create the impression that the application is hanging even when it is still working.

### Resolution
The app was validated by checking the live health endpoint and confirming the web interface was responding. This helped distinguish between a truly broken process and a slow inference pipeline that was still initializing.

In production or lower-powered environments, the following can help:

- use a lighter model or lower input size
- set lower resolution thresholds
- run only the necessary modules in the active pipeline
- reduce OCR frequency or disable plate recognition when not needed

---

## 7. The environment was sensitive to Python version and library compatibility

### Problem
A mismatch between Python and library compatibility can cause subtle runtime issues, especially with OpenCV or modern ML stacks.

### Resolution
The environment used for validation was the actively available Python 3.14 interpreter running the project successfully after dependency installation. The project configuration was updated to reflect the actual environment used during development rather than assuming a single standard interpreter setup.

---

## 8. Documentation needed to reflect the real project state, not the idealized structure

### Problem
The repository initially looked like a set of loosely connected scripts and modules, which made architecture understanding harder than it should be.

### Resolution
A structured documentation update was created to capture:

- the real working tree
- the active runtime entry points
- the main modules and responsibilities
- the dashboard flow
- the key challenges and successful fixes

This made onboarding much more straightforward for future runs, debugging, and feature work.

---

## Summary of the main wins

The major issues were resolved by:

1. launching the app from the correct working directory
2. installing the full project dependency set
3. verifying runtime health with the local API
4. confirming the app was serving the dashboard successfully
5. documenting the actual architecture and runtime flow

These steps transformed the project from a non-running codebase into a working local traffic analytics system.
