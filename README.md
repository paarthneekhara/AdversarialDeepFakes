# Adversarial DeepFakes

Deepfakes or facially manipulated videos, can be used maliciously to spread disinformation, harass individuals or defame famous personalities. Recently developed Deepfake detection methods rely on Convolutional Neural Network (CNN) based classifiers to distinguish AI-generated fake videos from real videos. In this work, we demonstrate that it is possible to bypass such detectors by adversarially modifying fake videos synthesized using existing Deepfake generation methods. We design adversarial examples for the FaceForensics++ dataset and fool victim CNN detectors - XceptionNet ([FaceForensics++ codebase](https://github.com/ondyari/FaceForensics) and [MesoNet](https://github.com/HongguLiu/MesoNet-Pytorch).

## Setup Requirements

The code is based on PyTorch and requires Python 3.6

Install requirements via ```pip install -r requirements.txt```

## Dataset
To download the FaceForensics++ dataset, you need to fill out their google form and, once accepted, they will send you the link to our download script.

Once, you obtain the download link, please head to the [download section of FaceForensics++](https://github.com/ondyari/FaceForensics/tree/master/dataset). You can also find details about the generation of the dataset there.

IMPORTANT: To create the test split of this dataset on which our experiments have been performed, use the script ```create_test_data.py``` as follows:
```python create_test_data.py --data_dir <path to DeepFakeDataset/manipulated_sequences> --dest_dir <path to DeepFakeDataset/manipulated_test_sequences> ```

Some sample fake videos have been uploaded here [GOOGLE DRIVE LINK]() to try out the attacks on a few examples. 

## Victim Pre-trained Models

### XceptionNet
The authors of FaceForensics++ provide XceptionNet model trained on our FaceForensics++ dataset. Besides the full image models, all models were trained on slightly enlarged face crops with a scale factor of 1.3.
You can find our used models under [this link](http://kaldir.vc.in.tum.de:/FaceForensics/models/faceforensics++_models.zip). Download them and save them in the root directory

### MesoNet

We use the the PyTorch implementation of [MesoNet](https://github.com/HongguLiu/MesoNet-Pytorch). The pretrained model can be downloaded from [here](https://github.com/HongguLiu/MesoNet-Pytorch/blob/master/output/Mesonet/best.pkl?raw=true). Once downloaded save the pkl as `Meso4_deepfake.pkl` inside ```faceforensics++_models_subset/face_detection/Meso```  directory which was created by unzipping the XceptionNet models in the previous link. 

### Detecting a video

```shell
python detect_from_video.py
-i <path to input video or folder of videos with extenstion '.mp4' or '.avi'>
-mi <path to pre-trained model weights >
-mt <type of model, choose either xception or meso >
-o <path to output folder, will contain output video(s) >
```  
Enable cuda with ```--cuda```  or see parameters with ```python detect_from_video.py -h```.

### Running an attack on video file

This setup is for running any of our attack methodologies to create adversarial examples on one video file. 
```shell
python attack.py
-i <path to input video or folder of videos with extenstion '.mp4' or '.avi'>
-mi <path to pre-trained model weights >
-mt <type of model, choose either xception or meso >
-o <path to output folder, will contain output video(s) >
-a <type of attack, choose from the following: robust, iterative_fgsm, black_box, black_box_robust >
--compress < if provided will save the adversarial video files in compressed MJPEG format > 

```  
Enable cuda with ```--cuda```  or see parameters with ```python attack.py -h```.

### Running experiments on FaceForensic++ Dataset

This setup is for running any of our attack methodologies to create adversarial examples on multiple video files from FaceForensic++ Dataset. 
```shell
python run_experiments.py
-data <path to manipulated_test_sequences folder containing all videos generated using Deepfakes, Neuraltextures, Face2Face, FaceSwap >
-mdir <path to pre-trained model weights >
-mtype <type of model, choose either xception or meso >
-exp <path to experiment directory, will contain output video(s) gnerated from the experiment>
-a <type of attack, choose from the following: robust, iterative_fgsm, black_box, black_box_robust >
--faketype <type of facial manipulation, choose from the following: Deepfakes, Neuraltextures, Face2Face, FaceSwap >
--compress < if provided will save the adversarial video files in compressed MJPEG format > 

```  
Enable cuda with ```--cuda```  or see parameters with ```python run_experiments.py -h```.

We run the following script in order to aggregate statistics on attack success rate from the experiments.
```shell
python aggregate_stats.py
-exp <path to experiment directory, contains video(s) gnerated from the experiment>
```  

# Requirements

- python 3.6
- requirements.txt
