mpiexec -n 8 python classifier_train.py --image_size 32 --log_interval 500  --data_dir data/cifar10_train/ &
python image_train.py --image_size 32 --log_interval 500 --lr_anneal_steps 100000  --data_dir ImageNet/ImageNet_train --class_cond True &
python classifier_sample.py --image_size 32 --class_cond True --model_path models/32x32_diffusion.pt --classifier_path models/32x32_classifier.pt &
