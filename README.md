# eo-detection-submit

[`[국방] 위성 이미지 객체 검출 대회`](https://dacon.io/competitions/official/235492/overview/) 의 알고리즘 제출을 위한 도커파일과 실행 스크립트 파일들입니다.



## Prerequistes

- Docker-CE >= 19.03.6 ([Install guide](https://docs.docker.com/install/linux/docker-ce/ubuntu/))
- Nvidia-Docker2 ([Install guide](https://github.com/NVIDIA/nvidia-docker))



## Usage

#### Installation & Build

```
$ git clone https://github.com/DeepBaksuVision/eo-detection-submit.git
$ sh build.sh
```



#### Setting

`[YOUR TRAIN DATASET DIR]`을 학습데이터셋이 있는 경로로 변경합니다.

```
$ vim run.sh
>>>
sudo docker run -i -t -d  --shm-size=16G \    
	-v [YOUR TRAIN DATASET DIR]:/root/Documents/dataset/trainset
        --gpus '"device=0"' \
        --name pengsoo_high pengsoo_high
```



#### Run & Execute

```
$ sh run.sh
$ docker exec -it pengsoo_high /bin/bash
```



> 자세한 실행방법은 첨부한 리포트 확인



