#!/bin/bash
export MASTER_PORT=8777
export MASTER_ADDR=localhost
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

echo $MASTER_PORT
echo $MASTER_ADDR
echo $WORLD_SIZE
echo $RANK
echo $LOCAL_RANK 

conda activate dino38
