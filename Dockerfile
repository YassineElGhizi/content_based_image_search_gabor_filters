FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN python3 -m pip install opencv-python

RUN python3 -m pip install glcontext

RUN python3 -m pip install PyOpenGL

RUN python3 -m pip install -r req.txt

CMD gunicorn --bind 0.0.0.0:5000 server:app