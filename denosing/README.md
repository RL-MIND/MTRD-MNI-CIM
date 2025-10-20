# MTRD-MNI-CIM for denosing




## How to run

### 1. Dependences
* [PyTorch](http://pytorch.org/)
* [torchvision](https://github.com/pytorch/vision)
* OpenCV for Python
* [HDF5 for Python](http://www.h5py.org/)
* [tensorboardX](https://github.com/lanpa/tensorboard-pytorch) (TensorBoard for PyTorch)

### 2. Train DnCNN-S (DnCNN with known noise level)
```
CUDA_VISIBLE_DEVICES=0 python train_kd_NC_bh.py --num_of_layers 17 --mode S --noiseL 25 --val_noiseL 25 --w_noiseL 0.3
CUDA_VISIBLE_DEVICES=2 python train_kd_pcm_NC_bh.py --num_of_layers 17 --mode S --noiseL 25 --val_noiseL 25 --w_noiseL 0.06
CUDA_VISIBLE_DEVICES=6 python train_kd_pcm_NC.py --num_of_layers 17 --mode S --noiseL 25 --val_noiseL 25 --w_noiseL 0.06
CUDA_VISIBLE_DEVICES=0 python train_nonid_pcm.py --num_of_layers 17 --mode S --noiseL 25 --val_noiseL 25 --w_noiseL 0.02
CUDA_VISIBLE_DEVICES=0 python train_nonid.py --num_of_layers 17 --mode S --noiseL 25 --val_noiseL 25 --w_noiseL 0.05
CUDA_VISIBLE_DEVICES=0 python train_CM.py --num_of_layers 17 --mode S --noiseL 25 --val_noiseL 25
```
**NOTE**
* If you've already built the training and validation dataset (i.e. train.h5 & val.h5 files), set *preprocess* to be False.
* According to the paper, DnCNN-S has 17 layers.
* *noiseL* is used for training and *val_noiseL* is used for validation. They should be set to the same value for unbiased validation. You can set whatever noise level you need.

### 3. Train DnCNN-B (DnCNN with blind noise level)
```
python train.py \
  --preprocess True \
  --num_of_layers 20 \
  --mode B \
  --val_noiseL 25
```
**NOTE**
* If you've already built the training and validation dataset (i.e. train.h5 & val.h5 files), set *preprocess* to be False.
* According to the paper, DnCNN-B has 20 layers.
* *noiseL* is ingnored when training DnCNN-B. You can set *val_noiseL* to whatever you need.

### 4. Test
```
python test.py --num_of_layers 17 --logdir logs/DnCNN-S-25 --test_data Set12 --test_noiseL 25
python test_RRAM_PCM_nonid.py --num_of_layers 17 --logdir logs/DnCNN-S-25 --test_data Set12 --test_noiseL 25
python test_RRAM_PCM-save2.py --num_of_layers 17 --logdir logs/DnCNN-S-25 --test_data Set12 --test_noiseL 25
python test_quant.py --num_of_layers 17 --logdir logs/DnCNN-S-25 --test_data Set12 --test_noiseL 25
```
**NOTE**
* Set *num_of_layers* to be 17 when testing DnCNN-S models. Set *num_of_layers* to be 20 when testing DnCNN-B model.
* *test_data* can be *Set12* or *Set68*.
* *test_noiseL* is used for testing. This should be set according to which model your want to test (i.e. *logdir*).



## Tricks useful for boosting performance
* Parameter initialization:  
Use *kaiming_normal* initialization for *Conv*; Pay attention to the initialization of *BatchNorm*
```
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)
```
* The definition of loss function  
Set *size_average* to be False when defining the loss function. When *size_average=True*, the **pixel-wise average** will be computed, but what we need is **sample-wise average**.
```
criterion = nn.MSELoss(size_average=False)
```
The computation of loss will be like:
```
loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
```
where we divide the sum over one batch of samples by *2N*, with *N* being # samples.
