FROM python:3

ENV PYTHONUNBUFFERED 1
RUN mkdir /src
WORKDIR /src

COPY requirements.txt /src/
RUN pip install -r requirements.txt
COPY deepcave/. /src/deepcave/
COPY server.py /src/
COPY worker.py /src/