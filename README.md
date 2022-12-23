# Perception-Validation-Verification

## Set Up Working Environment

### Get miniconda env for your OS of choice.

[miniconda for windows](https://docs.conda.io/en/latest/miniconda.html)

[miniconda for linux](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

Make sure conda is correctly installed and added to path by running.
```
conda --version 
```
### Create conda env

Make sure to create the environment for python 3.8 to avoid conflicts when installing pip modules later on.
```
conda create -n <env_name> python=3.8
```

### Install Pytorch with CUDA support

```
conda activate <env_name>
```
[Pytorch Previous Versions](https://pytorch.org/get-started/previous-versions/)

```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

### Install required modules

```
pip install -r requirements.txt
```


### Clone Repository & Build SMOKE Extension


Clone Repository in your folder of choice.
```
git clone https://github.com/HassanHotait/perception-validation-verification.git
```
Move to SMOKE dir
```
cd perception-validation-verification/SMOKE/
```
Build SMOKE extension
```
python setup.py build develop
```

### Download Data and Setup Repository Structure


``` 
perception-validation-verification
│──calibration_optimisation_data
│──camera_calibration
│──Dataset2Kitti    
│──results   
│──SMOKE
│    └──datasets
│──test_videos
│──tools   
└──YOLO_Detection   
```  

    