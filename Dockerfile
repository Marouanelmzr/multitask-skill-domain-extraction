FROM python:3.11-slim AS base

WORKDIR /app

RUN useradd --create-home --shell /usr/sbin/nologin appuser

FROM base AS development

COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y --no-install-recommends patchelf \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN find /usr/local/lib -name "onnxruntime_pybind11_state*.so" \
    -exec patchelf --clear-execstack {} \;

COPY scripts ./scripts
COPY models ./models
COPY data ./data

RUN sed -i 's/\r$//' ./scripts/download_models.sh \
    && chmod +x ./scripts/download_models.sh \
    && bash ./scripts/download_models.sh

EXPOSE 8000

CMD ["uvicorn", "scripts.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


FROM base AS production

COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y --no-install-recommends patchelf \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN find /usr/local/lib -name "onnxruntime_pybind11_state*.so" \
    -exec patchelf --clear-execstack {} \;

COPY scripts/main.py ./scripts/main.py
COPY scripts/inference.py ./scripts/inference.py
COPY scripts/normalise.py ./scripts/normalise.py
COPY scripts/download_models.sh ./scripts/download_models.sh
COPY models ./models
COPY data ./data

RUN sed -i 's/\r$//' ./scripts/download_models.sh \
    && chmod +x ./scripts/download_models.sh \
    && bash ./scripts/download_models.sh \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "scripts.main:app", "--host", "0.0.0.0", "--port", "8000"]