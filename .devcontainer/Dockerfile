FROM mcr.microsoft.com/devcontainers/python:3.13

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        graphviz \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspaces/RAG-LCC && chown vscode:vscode /workspaces/RAG-LCC

WORKDIR /workspaces/RAG-LCC

COPY --chown=vscode:vscode . .
