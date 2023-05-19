# IPEX_Dockerfile

- How to build

```
docker build ./ -f DockerFile.llm -t llm_centos8:latest
```

- docker build ./ -f DockerFile.llm -t llm_centos8:latest

```
docker build ./ --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${http_proxy} -f DockerFile.llm -t llm_centos8:latest
```
