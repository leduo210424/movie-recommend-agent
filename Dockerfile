FROM python:3.10-slim

# set workdir
WORKDIR /app

# avoid building caches
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# copy requirements first for caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . /app

# Expose port
EXPOSE 8000

# default command
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
