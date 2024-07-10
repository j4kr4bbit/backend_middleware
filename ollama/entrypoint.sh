#!/bin/bash

ollama serve &

sleep 10

ollama run llama3 &



curl -X POST http://localhost:11434/api/generate -d '{
  "model": "llama3",
  "prompt":"Why is the sky blue?",
  "format" : "json",
  "stream": false
 }'


 tail -f /dev/null
