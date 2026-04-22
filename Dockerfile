FROM python:3.11-slim AS base
WORKDIR /app

#pour le build : docker compose -f docker-compose.yml -f docker-compose.dev.yml
FROM base AS development
COPY requirements.txt .
COPY . .
EXPOSE 8000
RUN pip install -r requirements.dev.txt   
CMD ["uvicorn", "scripts.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

#pour le build : docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.ai.yml up --build
FROM base AS development-ai 
COPY requirements.ai.txt .
COPY . .
EXPOSE 8000  
RUN pip install -r requirements.ai.txt  
CMD ["uvicorn", "scripts.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

#pour le build : docker compose -f docker-compose.yml -f docker-compose.prod.yml
FROM base AS production
COPY requirements.dev.txt .
COPY . .
EXPOSE 8000
RUN pip install -r requirements.dev.txt
CMD ["uvicorn", "scripts.main:app", "--host", "0.0.0.0", "--port", "8000"]