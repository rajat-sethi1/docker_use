FROM python:3.7-slim

WORKDIR /app

RUN pip install -U pandas matplotlib numpy tensorflow scikit-learn keras kfp

# COPY preprocess.py ./preprocess.py

# ENTRYPOINT [ "python", "preprocess.py" ]