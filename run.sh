CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=23445 --use_env train_mobilenet.py --data-set IMNET --dist-eval --output output_path --batch-size 128 --model mobilenet --lr 0.1
