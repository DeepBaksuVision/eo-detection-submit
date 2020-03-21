# eo-detection-submit

[`[국방] 위성 이미지 객체 검출 대회`](https://dacon.io/competitions/official/235492/overview/) 의 알고리즘 제출을 위한 도커파일과 실행 스크립트 파일들로 구성되어있다.



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

`[YOUR TRAIN DATASET DIR]`과 `[YOUR TEST DATASET DIR]`을 학습 데이터셋과 테스트 데이터셋이 있는 경로로 변경한다.

```
$ vim run.sh
>>>
sudo docker run -i -t -d  --shm-size=16G \    
	-v [YOUR TRAIN DATASET DIR]:/root/Documents/dataset/trainset \
	-v [YOUR TEST DATASET DIR]:/root/Documents/dataset/testset \
    --gpus '"device=0"' \
    --name pengsoo_high pengsoo_high
```



학습 데이터셋과 테스트  데이터셋은 아래와 같이 구성된다.

```
.
|-- images
`-- label
```

- `images`: 학습 혹은 테스트 이미지가 있는 폴더. (Image는 `*.png`로 구성된다)
- label: 학습 혹은 테스트 라벨이 있는 폴더(label은 `*.json`으로 구성된다)



#### Run & Execute

```
$ sh run.sh
$ docker exec -it pengsoo_high /bin/bash
```



## Detail

### Structure

`root`폴더는 다음과 같이 구성되어있다.

```
root
`-- Documents
    |-- EO-Detection
    |-- dataset
    |-- detectron2
    |-- output
    `-- submit
```

- `EO-Detection`: 전처리/학습/Pretrained Model 추론을 실행할 수 있는 스크립트 파일 모음
- `dataset`: 학습데이터/전처리된 데이터가 저장되는 폴더
- `output`: 학습 시 tensorboard, config, checkpoint가 저장될 폴더
- `submit`: [국방] 위성 이미지 객체 검출 대회에 제출될 submit파일이 저장될 폴더



### Inference Using Pretrained Model

학습된 weights를 사용하여 테스트 데이터를 추론한다.

```
$ cd /root/Documents/EO-Detection/reproducing/script/pretrained_model_inference
$ sh inference.sh
```



추론이 완료되면 `/root/Documents/submit/final`에 최종 제출한 `submit.csv`파일을 확인할 수 있다.



### Reproducing Model

모델 재현을 위해서 다음과 같은 절차를 따른다.



#### Data pre-processing

`/root/Documents/EO-Detection/reproducing/script/make_dataset`에 있는 스크립트 파일을 차례로 실행한다.

```
$ cd /root/Documents/EO-Detection/reproducing/script/make_dataset
$ sh aircraft_sniper.sh
$ sh whole_sniper.sh
```



#### Train Model

데이터 전처리가 완료되었다면 차례로 전체 모델과 항공모함을 위한 별도의 모델을 학습한다.

**전체 모델 학습**

```
$ cd /root/Documents/EO-Detection/reproducing/script/model_train
$ sh train_whole_model.sh
```



**항공모함 모델 학습**

```
$ cd /root/Documents/EO-Detection/reproducing/script/model_train
$ sh train_aircraft_model.sh
```



#### Inference Using Reproduced Model

학습이 완료된 모델을 이용하여 테스트 데이터를 추론한다.

```
$ cd /root/Documents/EO-Detection/reproducing/script/trained_model_inference
$ sh inference.sh
```



추론이 완료되면 `/root/Documents/submit/final`에 최종 제출한 `submit.csv`파일을 확인할 수 있다.







