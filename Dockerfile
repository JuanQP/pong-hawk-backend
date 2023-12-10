# start from an official image
FROM python:3.11-slim-bullseye

# Create non-root user
RUN groupadd -g 1000 python
RUN useradd -m -s /bin/sh -u 1000 -g 1000 python

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.5.1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true

# System dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Folders
USER python
ENV WORKDIR /home/python/app
RUN mkdir -p ${WORKDIR}/assets
RUN mkdir -p ${WORKDIR}/static
RUN mkdir -p ${WORKDIR}/storage
RUN mkdir -p ${WORKDIR}/.venv/bin

WORKDIR ${WORKDIR}
ENV PATH="$WORKDIR/.venv/bin:/home/python/.local/bin:/usr/bin:$PATH"
RUN pip install "poetry==$POETRY_VERSION"

# Install dependencies
COPY pyproject.toml poetry.lock ${WORKDIR}/
RUN poetry install --no-interaction --no-ansi

# Add our code
COPY . ${WORKDIR}/

EXPOSE 8000
