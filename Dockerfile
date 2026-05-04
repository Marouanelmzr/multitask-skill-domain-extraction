FROM python:3.11-slim AS base
WORKDIR /app

#pour le build : docker compose -f docker-compose.yml -f docker-compose.dev.yml
FROM base AS development
COPY requirements.dev.txt .
#COPY . . : # COPY du code source en dernier : les dépendances sont ainsi mises en cache par Docker et pip install ne se réexécute que si requirements.txt change, pas à chaque modification du code.
EXPOSE 8000
RUN pip install --no-cache-dir -r requirements.dev.txt 
COPY scripts ./scripts
CMD ["uvicorn", "scripts.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

#pour le build : docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.ai.yml up --build
FROM base AS development-ai 
COPY requirements.ai.txt .
EXPOSE 8000  
RUN pip install --no-cache-dir -r requirements.ai.txt  
# we only copy the necessary files for the AI service to optimize the build cache. If we copied everything, any change in the code would invalidate the cache and require reinstalling all dependencies.
COPY scripts ./scripts
CMD ["uvicorn", "scripts.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

#pour le build : docker compose -f docker-compose.yml -f docker-compose.prod.yml
FROM base AS production
COPY requirements.dev.txt .
EXPOSE 8000
RUN pip install --no-cache-dir -r requirements.dev.txt
COPY scripts ./scripts
COPY src ./src
CMD ["uvicorn", "scripts.main:app", "--host", "0.0.0.0", "--port", "8000"]