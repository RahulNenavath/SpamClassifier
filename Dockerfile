FROM public.ecr.aws/lambda/python:3.8

COPY requirements.txt .

RUN /var/lang/bin/python3.8 -m pip install --upgrade pip

RUN pip --no-cache-dir install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
# Tensorflow 2.7.0 install from given path
#RUN python3.8 -m pip --no-cache-dir install \
#    https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.11.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

COPY src/ .

COPY src/main.py ${LAMBDA_TASK_ROOT}

CMD ["main.handler"]