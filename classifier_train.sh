TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 16 --lr 3e-4 --save_interval 10000 --log_interval 100 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
mpiexec -n 1 python classifier_train.py --data_dir ImageNet/ILSVRC/Data/CLS-LOC/train $TRAIN_FLAGS $CLASSIFIER_FLAGS 
