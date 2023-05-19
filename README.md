# LLM GPT-j (Large Language Model) based on Intel Extension For PyTorch
We provide the `Dockerfile.llm` used for building docker image with PyTorch, IPEX and Patched Transformers installed. The GPT-j model is using [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B).

## File structure
- `Dockerfile.llm`: The dockerfile used for dependencies installation.
- `run_gptj.json`: The inference benchmark script enabled ipex optimization path is used for measuring the performane of GPT-j.
- `prompt.json`: prompt.json for run_gptj.py
- `transformers.patch`: The patch file for transformers v4.26.1. This patch file also enabled our IPEX optimization for transformers and GPT-j.

## Build Docker image
- Option 1 (default): you could use `docker build` to build the docker image in your environment.
```
docker build ./ -f DockerFile.llm -t llm_centos8:latest
```

- Option 2: If you need to use proxy, please use the following command
```
docker build ./ --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${http_proxy} -f DockerFile.llm -t llm_centos8:latest
```

## Run GPT-j script
- **Step 1 Docker run**: We need to use `docker run` which helps us run our scirpt in docker image built previously.
```
# Docker run into docker image
docker run --privileged -v `pwd`:/root/workspace -it llm_centos8:latest
```

- **Step 2 Run GPT-j script**: At this step, we could activate our conda env and run GPT-j script with the following configuration: `max-new-tokens=32 num_beams=4`
```
# Activate conda env
source activate llm

# Run GPT-j workload
numactl -N 0 -m 0 python run_gptj.py --device cpu --ipex --jit --dtype bfloat16 --max-new-tokens 32
```
