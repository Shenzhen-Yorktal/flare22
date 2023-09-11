# Solution of Team aladdin5 for FLARE22 Challenge
This repository provides the solution of team aladdin5 for [MICCAI FLARE22 Challenge](https://flare22.grand-challenge.org).
For more information, please refer to https://flare22.grand-challenge.org.
The details of our method are described in our paper [Cascade Dual-decoders Network for Abdominal Organs Segmentation.](https://openreview.net/pdf?id=20WDOkjiyTu)
And our work are built upon [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), you can reproduce our method as follows step by step.  

## Environments and Requirements:
Firstly, please install [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) as below.  
```
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet  
pip install -e . 
```
You should meet the requirements of nnUNet, our method does not need any additional requirements. For more details, please refer to https://github.com/MIC-DKFZ/nnUNet. 

## 1. Generate Pseudo Labels
### 1.1 Dataset conversion
Following nnUNet, give a TaskID (e.g. Task201) to the 50 labeled data and organize them folowing the requirement of nnUNet.

```
put the Task201_Flare22AbdominalOrgansSegmentation.py into nnUNet\nnunet\dataset_conversion
then run Task201_Flare22AbdominalOrgansSegmentation.py

nnUNet_raw_data_base/nnUNet_raw_data/Task201_FLARE22labeled/
├── dataset.json
├── imagesTr
├── imagesTs
└── labelsTr
```

### 1.2 Data Preprocessing
Also, you should do some preprocessing. We use the default setting
```
nnUNet_plan_and_preprocess -t 22 -pl3d ExperimentPlanner3D_v21 -pl2d None
```
### 1.3 Training a basic model
Now, you can train the basic model. In our work, we trained with the all labeled data, also you can train with 5-fold.
```
run_training 3d_fullres nnUNetTrainerV2 -t 201 all -pl3d nnUNetPlansv2.1_plans_3D -pl2d None
```

### 1.4 Test on the Validation Set
The validation set has 50 CTs, which are provided by FLARE22. 
```
predict_simple -i valdir -o valpredict -t 201 -tr nnUNetTrainerV2  -m 3d_fullres  -p nnUNetPlansv2.1_plans_3D  --all_in_gpu True 
```
Then you can submit the validation result to FLARE22 and should get a pretty good mean DSC, our model is about 0.86. 
The actually score is not important.
### 1.5 Predict the 2000 unlabeled data
Then we generate the pseudo labels for the 2000 unlabeled CT scans. That is a really time-consuming thing. It takes 5 days to predict the all cases on 2 powerful AI platforms each with a NVIDIA RTX 3090 GPU. 
```
predict_simple -i unlabeleddir -o pseudolabeldir -t 201 -tr nnUNetTrainerV2  -m 3d_fullres  -p nnUNetPlansv2.1_plans_3D  --all_in_gpu True 
```
Now, we have 50 labeled cases and 2000 pseudo labeled cases. Then we can do supervised learning.

## 2. Cascade models 
As described in our paper, we design a cascade framework to reduce the memory consume, which consists of a low-resolution localization model and a high-resolution segmentation model.
You can download our pretrained models at [https://pan.baidu.com/s/14atMV7gAzN_P0zpaF8zGqA?pwd=37ik].
### 2.1 Localization model
 The localization model is a default nnUNet model but preprocessing the data in low-resolution, please refer to https://github.com/MIC-DKFZ/nnUNet. 
In short, take the 13 organs as foreground(labeled 1) and make a new dataset, then train a new model like the basic model in low-resolution space. 
During inference time, we can extract rois using this model.
This can be done by run extractROI, but do change the paths in the script.
```
extractROI 
```

### 2.2 Extract ROIs and build a new dataset
The abdomen is the ROI in this work. To train the new segmenation model, we should extract the ROIs and build a new abdomen dataset.
Also, just change the paths and run extractROI. Then you should do some data preprocess again!

```
extractROI
```
### 2.3 class-weighted Dual-decoders Segmentation model
We design a dual-decoders segmentation model and a class-weighted loss to improve the segmentation performance. In our original work, we did these in a gradually manner, however you can directly train a class-weighted segmentation model!
Your should do these things to train our model:

```
replace the dice_loss.py in nnUNet\nnunet\training\loss_functions with the provided dice_loss.py
put the nnUNetTrainerV2_MultiDecoder.py to nnUNet\nnunet\training\network_training
put the generic_modular_UNet_MultiDecoder.py to nnUNet\nnunet\network_architecture
```
Here nnUNet is the basedir of your nnUNet.
Then just train the model like the basic but with differenct trainer.
```
run_training 3d_fullres nnUNetTrainerV2_MultiDecoder -t taskid all -pl3d nnUNetPlansv2.1_plans_3D -pl2d None
```
Here taskid is assigned by yourself, we use 202.

### 2.4  post-processing 
Connected component-based post-processing is commonly used in medical image segmentation. Here we do post-processing for liver, kidneys, spleen and aorta . You should change the path in connected_components.py
```
connected_components
```
### 2.5 put the prediction back
Finally, you should put the prediction back to the original space.
Change the paths in put_roi_back.py and run it.
```
put_roi_back
```
## Others
Use predict_1by1.py and run_inference.py for docker only.

## Future
Our work is focus on the accurancy and we didn't modify nnunet source code for efficiency. A lot of optimization work can be done in the future.
