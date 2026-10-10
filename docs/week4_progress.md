# Week 4 Progress — API, Load Testing & Deployment

## Day 1-3: FastAPI Service
- Built REST API (`src/api.py`) with endpoints:
  - POST /detect — single image inference
  - POST /detect_batch — up to 20 images
  - GET /health — uptime and model status
  - GET /metrics — request counts, avg latency, defect stats
  - GET /docs — Swagger UI
- Swagger UI tested end-to-end with real defect images

## Day 4-5: Load Testing
- Async load test (`src/load_test.py`): 100 requests, 20 concurrent workers
- Results: 100% success rate, ~420 req/s, p95 18.7ms

## Day 6-7: Deployment
- Dockerised API (Dockerfile)
- Streamlit demo deployed to Streamlit Community Cloud
- README updated with setup guide and API docs
