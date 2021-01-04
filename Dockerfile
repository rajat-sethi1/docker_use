FROM python:3.7-slim

WORKDIR /app

RUN pip install -U pandas matplotlib scikit-learn numpy tensorflow keras kfp
