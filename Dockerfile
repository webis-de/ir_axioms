FROM openjdk:18-slim as openjdk-11

FROM python:3.9-slim as python

COPY --from=openjdk-11 /usr/local/openjdk-11 /usr/local/openjdk
ENV JAVA_HOME /usr/local/openjdk
RUN update-alternatives --install /usr/bin/java java /usr/local/openjdk/bin/java 1

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y && \
    apt-get install -y git build-essential

RUN --mount=type=cache,target=/root/.cache/pip \
    ([ -d /venv ] || python3.9 -m venv /venv) && \
    /venv/bin/pip install --upgrade pip

WORKDIR /workspace/

ADD pyproject.toml pyproject.toml
ARG PSEUDO_VERSION=1
RUN --mount=type=cache,target=/root/.cache/pip \
    SETUPTOOLS_SCM_PRETEND_VERSION=${PSEUDO_VERSION} \
    /venv/bin/pip install -e .[pyterrier]
RUN --mount=source=.git,target=.git,type=bind \
    --mount=type=cache,target=/root/.cache/pip \
    /venv/bin/pip install -e .[pyterrier]

ENV TERRIER_VERSION="5.7"
ENV TERRIER_HELPER_VERSION="0.0.7"
RUN /venv/bin/python -c "from pyterrier import init; init(version='${TERRIER_VERSION}', helper_version='${TERRIER_HELPER_VERSION}')"

ADD . .

ENTRYPOINT ["/venv/bin/python", "-m", "ir_axioms", "--offline"]
