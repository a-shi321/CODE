A PyTorch implementation of "Confused and Disentangled Distribution Alignment for Unsupervised Universal Adaptive Object Detection" (under review)



Our code is conducted based on [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch), please setup the framework by it.

# Preparation
## Install Pytorch

 Our code is conducted based on [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch),please setup the framework by it.

## Download dataset

we use cityscape and cityscapes-foggy datasets respectly as source and target,the cityscapes dataset could be download [Here](https://www.cityscapes-dataset.com/login/)

the format of datasets is similar with VOC,you just need to split train.txt to train_s.txt and train_t.txt


## Train and Test
1.train the model,you need to download the pretrained model [vgg_caffe](https://github.com/jwyang/faster-rcnn.pytorch） which is different with pure pytorch pretrained model

2.change the dataset root path in ./lib/model/utils/config.py and some dataset dir path in ./lib/datasets/cityscape.py,the default data path is ./data

3 Train the model

### train cityscapes -> cityscapes-foggy
CUDA_VISIBLE_DEVICES=GPU_ID python da_trainval_net.py --dataset cityscape --net vgg16 --bs 1 --lr 1e-3 --lr_decay_step 6 --cuda

### Test model in target domain 
CUDA_VISIBLE_DEVICES=GPU_ID python test.py --dataset cityscape --part test_t --model_dir=# The path of your pth model --cuda

