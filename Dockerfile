# syntax=docker/dockerfile:1

FROM continuumio/miniconda3

# Install linux dependencies
RUN apt-get update -y
RUN apt install -y build-essential
RUN apt-get install -y swig
RUN apt-get install -y redis-server

# Copy files
COPY . /DeepCAVE
WORKDIR /DeepCAVE

RUN conda update conda -y

# Create new environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "DeepCAVE", "/bin/bash", "-c"]

# Install DeepCAVE
RUN pip install .