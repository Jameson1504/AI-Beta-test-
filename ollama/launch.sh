#!/usr/bin/env bash
# Launches Ollama and optionally pulls/runs a specified model.
set -euo pipefail

MODEL="${1:-llama3.2}"
HOST="${OLLAMA_HOST:-127.0.0.1}"
PORT="${OLLAMA_PORT:-11434}"

export OLLAMA_HOST="${HOST}:${PORT}"

check_ollama() {
  if ! command -v ollama &>/dev/null; then
    echo "Ollama not found. Install it from https://ollama.com/download"
    exit 1
  fi
}

start_server() {
  if curl -sf "http://${HOST}:${PORT}/api/tags" &>/dev/null; then
    echo "Ollama already running on ${HOST}:${PORT}"
    return 0
  fi
  echo "Starting Ollama server on ${HOST}:${PORT}..."
  ollama serve &
  OLLAMA_PID=$!
  echo "Ollama PID: ${OLLAMA_PID}"

  for i in $(seq 1 20); do
    if curl -sf "http://${HOST}:${PORT}/api/tags" &>/dev/null; then
      echo "Ollama is ready."
      return 0
    fi
    sleep 0.5
  done
  echo "Timed out waiting for Ollama to start."
  exit 1
}

pull_model() {
  local model="$1"
  echo "Pulling model: ${model}"
  ollama pull "${model}"
}

run_model() {
  local model="$1"
  echo "Running model: ${model}"
  ollama run "${model}"
}

main() {
  check_ollama
  start_server
  pull_model "${MODEL}"
  run_model "${MODEL}"
}

main
