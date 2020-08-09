#!/bin/sh

steps=7

#/opt/miniconda3/envs/DT-RAM/bin/python main.py --use_gpu=False --num_workers=4 --num_glimpses=1 --ckpt_dir=ckpt_mnist

#for((i=5;i<=$steps;i++))
#do
#echo "training ""$i"" step"
#/opt/miniconda3/envs/DT-RAM/bin/python main.py --use_gpu=False --num_workers=4 --num_glimpses=$i --resume=True --resume_fresh=True --ckpt_dir=ckpt_mnist --ckpt_name="ram_""$((i-1))""_8x8_1_model_best.pth.tar"
#done

#echo "training dynamic DT-RAM1"
#/opt/miniconda3/envs/DT-RAM/bin/python main.py --use_gpu=False --num_workers=4 --train_dynamic=True --resume=True --resume_fresh=True --ckpt_dir=ckpt_mnist --ckpt_name="ram_""${steps}""_8x8_1_model_best.pth.tar" --num_glimpses=$steps --gamma=0.98 --batch_size=20 --init_lr=0.001 --momentum=0.9 --std=0.11

echo "training dynamic DT-RAM2"
/opt/miniconda3/envs/DT-RAM/bin/python main.py --use_gpu=False --num_workers=4 --train_dynamic=True --resume=True --resume_fresh=True --ckpt_dir=ckpt_mnist --ckpt_name="ram_""${steps}""_8x8_1_model_best.pth.tar" --num_glimpses=$steps --gamma=0.99  --batch_size=20 --init_lr=0.001 --momentum=0.9 --std=0.11
