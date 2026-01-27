
FROM python:3.11

WORKDIR /app

COPY jules_session_7226522193343951134 /app/jules_session_7226522193343951134
COPY models /app/models
COPY v1 /app/v1

RUN pip install --no-cache-dir -r /app/jules_session_7226522193343951134/requirements.txt

CMD ["python", "/app/jules_session_7226522193343951134/direct_training.py"]
