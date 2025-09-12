FROM python:3.11.9-slim

WORKDIR /app/backend

RUN pip install --upgrade pip

COPY backend/requirements.txt .
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

COPY backend/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]