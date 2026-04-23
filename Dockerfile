FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install python dependencies FIRST (layer caching)
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project AFTER dependencies (so code changes don't rebuild pip layer)
COPY . /app/

# Collect static files
RUN python manage.py collectstatic --noinput

# Railway assigns PORT dynamically
CMD gunicorn analytic_agent.wsgi:application --bind 0.0.0.0:$PORT
