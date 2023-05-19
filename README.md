# IPEX_Dockerfile

- How to build

```
docker build ./ -f DockerFile.llm -t llm_centos8:latest
```

- If you need to use proxy, please use the following command

```
docker build ./ --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${http_proxy} -f DockerFile.llm -t llm_centos8:latest
```

- How to Run
```
# Docker run into docker image
docker run -v ${WORKSPACE}:/root/workspace -it llm_centos8:latest

# Activate conda env
source activate llm
```