FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install flask joblib scikit-learn pillow numpy

EXPOSE 5000

CMD ["python", "app.py"]