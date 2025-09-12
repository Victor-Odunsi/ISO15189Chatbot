FROM python:3.11.9-slim

WORKDIR /app/frontend

COPY frontend/requirements.txt .
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

COPY frontend/ .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]