FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN \
    --mount=type=cache,target=/var/cache/apt \
    apt-get update -y && \
    apt-get upgrade -y
RUN \
    --mount=type=cache,target=/var/cache/apt \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa
RUN \
    --mount=type=cache,target=/var/cache/apt \
    apt-get install -y python3.9 python3.9-dev python3.9-venv python3-pip python-is-python3 openjdk-11-jdk git

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/

RUN \
    --mount=type=cache,target=/root/.cache/pip \
    ([ -d /venv ] || python3.9 -m venv /venv) && \
    /venv/bin/pip3 install --upgrade pip

WORKDIR /workspace/
ADD .git/ .git/
ADD pyproject.toml pyproject.toml

RUN \
    --mount=type=cache,target=/root/.cache/pip \
    /venv/bin/pip3 install /workspace[pyterrier]

ADD . .

ENTRYPOINT ["/venv/bin/python3.9", "-m", "ir_axioms"]