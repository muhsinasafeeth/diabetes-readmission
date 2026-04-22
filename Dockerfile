# ── Stage 1: dependency builder ───────────────────────────────────────────────
# Uses a full Python image to compile wheels for any C-extension packages.
# The compiled wheels are then copied into the slim final image, keeping it small.
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed to compile packages like xgboost / scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt


# ── Stage 2: final production image ───────────────────────────────────────────
FROM python:3.11-slim

# Create a non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Copy pre-built wheels from builder stage and install (no compilation needed)
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels /wheels/* \
 && rm -rf /wheels

# Copy application source
COPY app.py          .
COPY templates/      ./templates/
COPY static/         ./static/

# Copy trained model artefacts
# Expected files inside ./models/:
#   xgb_model.pkl, scaler_standard.pkl, scaler_robust.pkl, feature_columns.pkl
COPY models/         ./models/

# Hand ownership to the non-root user
RUN chown -R appuser:appgroup /app
USER appuser

# Expose the port Gunicorn will listen on
EXPOSE 5000

# Health check — Docker will mark the container unhealthy if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

# Run with Gunicorn (4 workers). For CPU-bound ML inference, 2–4 workers is typical.
# Adjust --workers based on available CPU cores: (2 × cores) + 1 is a common formula.
CMD ["gunicorn", \
     "--bind",    "0.0.0.0:5000", \
     "--workers", "4", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile",  "-", \
     "app:app"]
