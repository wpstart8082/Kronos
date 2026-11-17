# Kronos prediction service (GPU).
# Base ships torch with CUDA runtime - host only needs the NVIDIA driver.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# install python deps first for better layer caching
COPY kronos_service/requirements.txt /app/kronos_service/requirements.txt
RUN pip install --no-cache-dir -r /app/kronos_service/requirements.txt

# application code (model package + service + smoke-test data)
COPY model /app/model
COPY kronos_service /app/kronos_service
COPY tests/data /app/tests/data

# non-root user; writable dirs for SQLite/logs
RUN useradd -m appuser \
    && mkdir -p /app/instance /app/logs /models \
    && chown -R appuser:appuser /app/instance /app/logs /app/kronos_service
USER appuser

EXPOSE 8000

WORKDIR /app/kronos_service
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
