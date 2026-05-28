FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/audio_files data/models

EXPOSE 5000

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["python", "web_server.py"]
