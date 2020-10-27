FROM tensorflow/tensorflow:latest-gpu

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

ENV PYTHONPATH=/biomag
WORKDIR /biomag