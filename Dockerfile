FROM nvcr.io/nvidia/tensorflow:20.09-tf2-py3

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

ENV PYTHONPATH=/biomag
WORKDIR /biomag
