FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY models/ ./models/
COPY projects/student_prediction/ ./projects/student_prediction/
EXPOSE 8000
CMD ["uvicorn", "projects.student_prediction.fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]
