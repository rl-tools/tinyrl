FROM ubuntu:latest
ARG TINYRL_USE_CLANG=false

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip 

RUN if [ "$TINYRL_USE_CLANG" = "true" ]; then \
        sudo apt-get update && sudo apt-get install -y clang; \
    fi

RUN pip3 install --upgrade pip && pip3 install \
    pytest \
    gymnasium mkl mkl-include