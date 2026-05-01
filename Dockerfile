FROM python:3.11-slim

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY project/ ./project/
COPY dataset/ ./dataset/

EXPOSE 8000

CMD ["python", "-m", "project.api.run_api"]
