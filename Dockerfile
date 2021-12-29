FROM public.ecr.aws/lambda/python:3.8

COPY requirements.txt .

RUN /var/lang/bin/python3.8 -m pip install --upgrade pip

RUN pip --no-cache-dir install -r requirements.txt
# Tensorflow 2.7.0 install from given path
RUN pip --no-cache-dir install \
    https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.7.0-cp38-cp38-manylinux2010_x86_64.whl

COPY src/ .

#CMD ["python", "main.py"]
CMD ["main.handler"]