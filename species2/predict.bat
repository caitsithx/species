echo 'This predicting process takes around 24 hours, please be patient and do not interrupt it...'

REM python predict.py --tta resnet34
python predict.py --tta resnet50 
python predict.py --tta resnet101 
python predict.py --tta resnet152 

python predict.py --tta densenet121 
REM python predict.py --tta densenet161
python predict.py --tta densenet169 
python predict.py --tta densenet201 

python predict.py --tta inception_v3 

python predict.py --tta vgg19_bn 
python predict.py --tta vgg16_bn 


echo 'predicting finished.'
