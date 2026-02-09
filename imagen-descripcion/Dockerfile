FROM python:3.11-slim

WORKDIR /app

# Copiamos solo lo que tenemos
COPY requirements.txt .
COPY streamlit_app.py .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
