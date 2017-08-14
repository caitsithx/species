echo 'This training process takes around 24 hours, please be patient and do not interrupt it...'

python train.py --fine_tune resnet34 70
python train.py --fine_tune resnet50 70
python train.py --fine_tune resnet101 70
python train.py --fine_tune resnet152 70

python train.py --fine_tune densenet121 70
python train.py --fine_tune densenet161 70
python train.py --fine_tune densenet169 70
python train.py --fine_tune densenet201 70

python train.py --fine_tune inception_v3 70

python train.py --fine_tune vgg19_bn 70
python train.py --fine_tune vgg16_bn 70


echo 'Training finished.'
