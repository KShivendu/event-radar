FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 EVENT_RADAR_DATA=/data
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Default: the web UI. compose.yml runs the same image as the scraper too.
EXPOSE 5050
CMD ["python", "-m", "flask", "--app", "ui", "run", "--host", "0.0.0.0", "--port", "5050"]
