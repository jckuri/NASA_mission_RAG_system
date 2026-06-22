#!/bin/sh

set -e

if [ ! -f .env ]; then
  echo "Error: .env file not found"
  exit 1
fi

set -a
. ./.env
set +a

python embedding_pipeline.py --openai-key "$OPENAI_API_KEY" --data-path "data_text"