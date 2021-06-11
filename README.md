# Adversarial Deepfakes

Deepfakes or facially manipulated videos, can be used maliciously to spread disinformation, harass individuals or defame famous personalities. Recently developed Deepfake detection methods rely on Convolutional Neural Network (CNN) based classifiers to distinguish AI-generated fake videos from real videos. In this work, we demonstrate that it is possible to bypass such detectors by adversarially modifying fake videos synthesized using existing Deepfake generation methods. We design adversarial examples for the FaceForensics++ dataset and fool victim CNN detectors - XceptionNet ([FaceForensics++ codebase](https://github.com/ondyari/FaceForensics)) and [MesoNet](https://github.com/HongguLiu/MesoNet-Pytorch).


## Setup Requirements

The code is based on PyTorch and requires Python 3.6

Install requirements via ```pip install -r requirements.txt```

## Dataset
To download the FaceForensics++ dataset, you need to fill out their google form and and once accepted, they will send you the link to the download script.

Once, you obtain the download link, please head to the [download section of FaceForensics++](https://github.com/ondyari/FaceForensics/tree/master/dataset). You can also find details about the generation of the dataset there. To reproduce the experiment results in this paper, you only need to download the c23 videos of all the fake video generation methods.

IMPORTANT: To create the test split of this dataset on which our experiments have been performed, use the script ```create_test_data.py``` as follows:
```python create_test_data.py --data_dir <path to DeepFakeDataset/manipulated_sequences> --dest_dir <path to DeepFakeDataset/manipulated_test_sequences> ```

## Small subset to try out the attacks
If you want to try out the attacks on a very small subset of this dataset, create a directory `Data/` in the root folder,  download the zipped dataset from [this link](http://adversarialdeepfakes.github.io/dfsubset.zip) and save unzip inside `Data/` to have the following directory structure:

```
Data/
  - DFWebsite/
      - Deepfakes/
      - Face2Face/
      - FaceSwap/
      - NeuralTextures/
```

## Victim Pre-trained Models

### XceptionNet
The authors of FaceForensics++ provide XceptionNet model trained on our FaceForensics++ dataset. 
You can find our used models under [this link](http://kaldir.vc.in.tum.de:/FaceForensics/models/faceforensics++_models.zip). Download zip and unzip it in the root project directory.

### MesoNet

We use the the PyTorch implementation of [MesoNet](https://github.com/HongguLiu/MesoNet-Pytorch). The pretrained model can be downloaded from [here](https://github.com/HongguLiu/MesoNet-Pytorch/blob/master/output/Mesonet/best.pkl?raw=true). Once downloaded save the pkl as `Meso4_deepfake.pkl` inside ```faceforensics++_models_subset/face_detection/Meso```  directory which was created by unzipping the XceptionNet models in the previous link. 

After saving the weights the `faceforensics++_models_subset/` directory should have the following structure:

```
faceforensics++_models_subset/
  - face_detection/
    - Meso
      - Meso4_deepfake.pkl
    - xception
      - all_c23.p
```
    

### Detecting a video

```shell
python detect_from_video.py
-i <path to input video or folder of videos with extenstion '.mp4' or '.avi'>
-mi <path to pre-trained model weights >
-mt <type of model, choose either xception or meso >
-o  <path to output folder, will contain output video(s) >
```
Enable cuda with ```--cuda```  or see parameters with ```python attack.py -h```.

Example:
```shell
python detect_from_video.py -i Data/DFWebsite/Face2Face/c23/videos/183_253.mp4 -mi faceforensics++_models_subset/xception/all_c23.p -mt xception -o tempout/ --cuda
```

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

Enable cuda with ```--cuda```  or see parameters with ```python detect_from_video.py -h```.

Example:
```shell
python attack.py -i Data/DFWebsite/Face2Face/c23/videos/183_253.mp4 -mi faceforensics++_models_subset/xception/all_c23.p -mt xception -o temadv/ -a robust --cuda --compress
```

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

Example for running the experiments on the small subset provided:

```shell
python run_experiments.py -data Data/DFWebsite -mdir faceforensics++_models_subset -mtype xception -exp ExpTemp -a robust --faketype Face2Face --compress
```

### Citing this work
If you use Adversarial Deepfakes for academic research, you are highly encouraged to cite the following papers:

1) https://openaccess.thecvf.com/content/WACV2021/html/Hussain_Adversarial_Deepfakes_Evaluating_Vulnerability_of_Deepfake_Detectors_to_Adversarial_Examples_WACV_2021_paper.html

@InProceedings{Hussain_2021_WACV,
    author    = {Hussain, Shehzeen and Neekhara, Paarth and Jere, Malhar and Koushanfar, Farinaz and McAuley, Julian},
    title     = {Adversarial Deepfakes: Evaluating Vulnerability of Deepfake Detectors to Adversarial Examples},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2021},
    pages     = {3348-3357}
}

2) https://dl.acm.org/doi/10.1145/3464307
@article{10.1145/3464307,
author = {Hussain, Shehzeen and Neekhara, Paarth and Dolhansky, Brian and Bitton, Joanna and Canton Ferrer, Cristian and McAuley, Julian and Koushanfar, Farinaz},
title = {Exposing Vulnerabilities of Deepfake Detection Systems with Robust Attacks},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {0},
number = {ja},
issn = {2692-1626},
url = {https://doi.org/10.1145/3464307},
doi = {10.1145/3464307},
journal = {Digital Threats: Research and Practice}
}


