FROM python:3.8.2-slim-buster

WORKDIR /app/src

COPY requirements.txt .

RUN /usr/local/bin/python -m pip install --upgrade pip

RUN apt-get update && apt-get -y install libpq-dev gcc

RUN pip install -r requirements.txt

COPY src/ .

CMD ["python", "main.py"]