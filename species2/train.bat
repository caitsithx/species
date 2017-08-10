echo 'This training process takes around 24 hours, please be patient and do not interrupt it...'

REM python train.py --train inception_v3
REM python train.py --fine_tune inception_v3
REM python train.py --train vgg19_bn
REM python train.py --fine_tune vgg19_bn
REM python train.py --train vgg16_bn
REM python train.py --fine_tune vgg16_bn
REM python train.py --train densenet169
REM python train.py --fine_tune densenet169
REM python train.py --train densenet161
REM python train.py --fine_tune densenet161

python train.py --pseudo resnet50 submit-20170804-095211.csv
python train.py --pseudo inception_v3 submit-20170804-095211.csv
python train.py --pseudo vgg19_bn submit-20170804-095211.csv
python train.py --pseudo vgg16_bn submit-20170804-095211.csv
python train.py --pseudo densenet169 submit-20170804-095211.csv
python train.py --pseudo densenet121 submit-20170804-095211.csv
python train.py --pseudo densenet201 submit-20170804-095211.csv
python train.py --pseudo resnet101 submit-20170804-095211.csv
python train.py --pseudo resnet152 submit-20170804-095211.csv

echo 'Training finished.'
