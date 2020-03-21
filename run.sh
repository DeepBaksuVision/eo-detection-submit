sudo docker run -i -t -d  --shm-size=16G \
	-v [YOUR TRAIN DATASET DIR]:/root/Documents/dataset/trainset
        --gpus '"device=0"' \
        --name pengsoo_high pengsoo_high
