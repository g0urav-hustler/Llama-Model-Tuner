


<h1  align="center"  > LLama Model Tuner </h1>

  
  

## Overview

#### This tool helps to fine tune llama model on your custom dataset .
  

----------------------------

  

## Tool Process

  

1. Upload your custom dataset in csv format.
![Webui Image](https://github.com/g0urav-hustler/Llama-Model-Tuner/blob/main/readme_sources/photo_1.png)
  

2. Select the parameter and copy your model name from hugging face model web.
![Webui Image](https://github.com/g0urav-hustler/Llama-Model-Tuner/blob/main/readme_sources/photo_2.png)
  

3. Train the model and see model result.
![Webui Image](https://github.com/g0urav-hustler/Llama-Model-Tuner/blob/main/readme_sources/photo_3.png)

4. See model result and download the model as zip file.
![Webui Image](https://github.com/g0urav-hustler/Llama-Model-Tuner/blob/main/readme_sources/photo_4.png)

6. Try different model outputs mannually
![Webui Image](https://github.com/g0urav-hustler/Llama-Model-Tuner/blob/main/readme_sources/photo_5.png)

  
## Project Pipeline

  
Pipeline of the Project:

  
  

1. Data Ingestion

  

2. Data Processing

  

3. Training Model

  

4. Model Evaluation

  

5. Webui

----------------------------
## Dockerization 

Use this tool as docket container .


Pre-requisite: Docker  installed on your computer.

To install Docker, see [Reference](https://runnable.com/docker/getting-started/)

Login to your docker account 
```
$ docker login
```

Docker command to build the docker container
```
$ docker build -t [tool_name]:latest .
```
Pushing the docker container on the hub
```
$ docker push 
```