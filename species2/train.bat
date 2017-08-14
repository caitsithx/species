echo 'This training process takes around 24 hours, please be patient and do not interrupt it...'

python train.py --train resnet34 30
python train.py --train resnet50 30
python train.py --train resnet101 30
python train.py --train resnet152 30

python train.py --train densenet121 30
python train.py --train densenet161 30
python train.py --train densenet169 30
python train.py --train densenet201 30

python train.py --train inception_v3 30

python train.py --train vgg19_bn 30
python train.py --train vgg16_bn 30


echo 'Training finished.'
