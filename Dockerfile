FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/
RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY src/ /app/src/
ENTRYPOINT ["python", "/app/src/train.py"]
