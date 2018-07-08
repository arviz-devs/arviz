FROM jupyter/minimal-notebook

ARG SRC_DIR

MAINTAINER Austin Rochford <austin.rochford@gmail.com>

ADD $SRC_DIR /home/jovyan/
RUN /bin/bash /home/jovyan/scripts/create_testenv.sh

ENV PYTHONPATH $PYTHONPATH:"$HOME"
