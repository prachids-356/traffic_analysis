# Project Roadmap and Future Improvements

This roadmap captures the next logical improvements for the traffic analytics project based on its current working state and the functionality already implemented.

## Current status

The project is operating as a functional local traffic analytics dashboard with:

- YOLO-based vehicle detection
- tracking across frames
- vehicle classification
- speed distribution analysis
- lane and flow monitoring
- recent alert logs
- live dashboard rendering via FastAPI

---

## Near-term improvements

### 1. Better test coverage

Add automated tests for:

- route endpoints
- health checks
- video upload flow
- stats aggregation
- database insert and fetch functions

This will reduce regressions and help confirm the app is working after dependency updates or code refactors.

### 2. Cleaner configuration management

Introduce a more structured config layer with:

- environment-based settings
- defaults for local and production modes
- explicit camera source settings
- separation of model-specific and runtime-specific configuration

This makes the app easier to run across multiple environments.

### 3. More robust model tuning

Improve detection performance with:

- fine-tuned thresholds
- lane-specific filtering
- region-of-interest tuning
- confidence calibration for different camera angles

The current implementation is strong for a demo, but model tuning would improve reliability in real traffic conditions.

---

## Medium-term improvements

### 4. Real-time analytics dashboard improvements

Enhance the dashboard with:

- trend charts over time
- more detailed lane analytics
- real-time heatmap updates
- downloadable logs and summaries
- richer alert classification

This would make the dashboard more useful for operators and analysts.

### 5. Better event handling and alerting

Expand alert logic to include:

- high-vehicle-density alerts
- suspicious stop patterns
- wrong-direction logic per road segment
- queue length monitoring
- lane blockage detection

### 6. Database modernization

For larger deployments, evolve from local storage into:

- PostgreSQL or MySQL
- time-series storage for traffic data
- scheduled cleanup jobs
- event indexing and analytics queries

---

## Long-term opportunities

### 7. Multi-camera support

Expand the backend to support:

- multiple live streams
- centralized aggregation across intersections
- per-camera dashboard tiles
- distributed processing for larger networks

### 8. Edge deployment and scalability

Move toward:

- GPU-accelerated inference
- asynchronous workers
- container orchestration
- API service separation from processing tasks

### 9. Smart city integration

Connect the platform to broader transportation systems, such as:

- traffic signal optimization
- congestion forecasting
- historical route analysis
- incident detection pipelines

---

## Recommended next milestone

The best next step is to build a robust test suite and improve configuration management. Once those are in place, the project can expand more safely into multi-camera and production-style deployment.

This sequence keeps the project stable while enabling future analytics and system-scale upgrades.
