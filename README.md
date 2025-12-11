# LPATR-Net

Here is the source code for the Pytorch implementation of our paper:

**LPATR-Net: Learnable Piecewise Affine Transformation Regression Assisted Data-Driven Dehazing Framework**

 We have appended all training codes to ensure the training process could be easily conducted.
 
 Supplementary materials are available at https://www.alipan.com/s/B396Ro82EQA
 
**The repository includes:**

1. Dependencies.
1. Data Preparation.
1. Training code.
2. Testing code.
2. Application code.

## Dependencies

```python
Ubuntu 20.04
python 3.8
pytorch 1.9.1
torchvision 0.10.1 
torchaudio 0.9.1
opencv-python 4.7.0.68
numpy 1.24.1
pytorch-msssim 1.0.0
warmup_scheduler 0.3
```

## **Data Preparation**

Apologies for being unable to provide all the relevant image datasets due to size constraints. Nevertheless, all the datasets referred to in the paper are publicly available and can be downloaded with ease.

**Dataset Organization Form**

For training and testing purposes, your directory architecture should be laid out according to the sample below .

```
|--dataset
    |--reside-indoor 
        |--train
        	|--hazy
        		|--1_1_0.90179.png
            		:
        	|--gt
        		|--1.png
            		:
        |--test
        	|--hazy
        		|--1400_1.png
            		:
        	|--gt
        		|--1400.png
            		:
    |--reside-outdoor 
        |--train
        	|--hazy
        		|--0001_0.8_0.1.jpg
            		:
        	|--gt
        		|--0001.png
            		:
        |--test
        	|--hazy
        		|--0001_0.8_0.2.jpg
            		:
        	|--gt
        		|--0001.png
            		:
    |--O-HAZE
        |--train
        	|--hazy
        		|--01_outdoor_hazy.jpg
            		:
        	|--gt
        		|--01_outdoor_GT.jpg
            		:
        |--test
        	|--hazy
        		|--03_outdoor_hazy.jpg
            		:
        	|--gt
        		|--03_outdoor_GT.jpg
            		:
```

## Training

The checkpoint will be saved at `./results/LPATRNet/{DatasetName}/checkpoint`, and the log will be saved at `./results/LPATRNet/{DatasetName}/log`.

**Training on Reside-outdoor**

```shell
$ python main.py --mode train --batch_size 4 --learning_rate 1e-4 --num_epoch 50 --DatasetName reside-outdoor --data_dir {The path to reside-outdoor dataset} 
```
**Training on Reside-indoor**

```shell
$ python main.py --mode train --batch_size 4 --learning_rate 1e-4 --num_epoch 1000 --DatasetName reside-indoor --data_dir {The path to reside-indoor dataset} 
```
**Training on O-HAZE**

```shell
$ python main.py --mode train --batch_size 4 --learning_rate 1e-4 --num_epoch 500 --DatasetName O-HAZE --data_dir {The path to O-HAZE dataset} 
```

## Testing

The output will be saved at `./results/LPATRNet/{DatasetName}/output`

**Testing on Reside-outdoor**

```shell
$ python main.py --mode test --DatasetName reside-outdoor --data_dir {The path to reside-outdoor dataset} --pre_model_path {The path to the pre-trained model}
```

**Testing on Reside-indoor**

```shell
$ python main.py --mode test --DatasetName reside-indoor --data_dir	{The path to reside-indoor dataset} --pre_model_path {The path to the pre-trained model}
```

**Testing on O-HAZE**

```shell
$ python main.py --mode test --DatasetName O-HAZE --data_dir {The path to O-HAZE dataset} --pre_model_path {The path to the pre-trained model}
```

## **Application**

If you want to test the effect on a single image,  run the following code.

```bash
$ python test_single_image.py   --pre_model_path {The path to the pre-trained model} --input_file_path {The path to a hazy picture}  --output_file_path {The path to save the dehaze image}
```

