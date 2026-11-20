# Container that serves the web app + API (built by Render on every push)
FROM python:3.12-slim

WORKDIR /app

# 1. Install only the packages needed to serve predictions
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 2. Copy the app and the trained models (no training code, no data)
COPY src/api.py src/
COPY src/static/ src/static/
COPY models/ models/

# 3. Start the server for the outside world (Render provides the port number)
CMD ["sh", "-c", "uvicorn api:app --app-dir src --host 0.0.0.0 --port ${PORT:-10000}"]
