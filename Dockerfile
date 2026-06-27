FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment variables
ENV PYTHONPATH=/app

EXPOSE 8000
EXPOSE 8501

CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port ${PORT:-8501} --server.address 0.0.0.0"]
