#!/bin/bash
# Launch vLLM inference endpoint
huggingface-cli login --token $HUGGINGFACE_TOKEN
nohup python -m vllm.entrypoints.openai.api_server --model $LLM_MODEL > vllm.out 2>vllm.err &
( tail -f -n0 vllm.err & ) | grep -q "Uvicorn running" & jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' &
tail -f /dev/null
