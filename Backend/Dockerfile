FROM python:3.8.12-slim

WORKDIR /app/src

COPY requirements.txt .

RUN /usr/local/bin/python -m pip install --upgrade pip && apt-get update && apt-get -y install libpq-dev gcc

RUN pip --no-cache-dir install -r requirements.txt

RUN pip --no-cache-dir install \
    https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.7.0-cp38-cp38-manylinux2010_x86_64.whl

COPY src/ .

CMD ["python", "main.py"]
