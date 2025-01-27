# CURE
Towards Universal Certified Robustness with Multi-Norm Training [[Arxiv]](https://arxiv.org/abs/2410.03000)\
*Enyi Jiang, David S. Cheung, Gagandeep Singh*

### Setup
Create and activate a conda environment
```
conda create --name CURE python=3.10.4
conda activate CURE
```
Install the requirements
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

Add the main directory to the PYTHONPATH (make sure you are in the top level directory)
```
export PYTHONPATH=$PWD:$PYTHONPATH
```

To download the TinyImageNet dataset navigate into the `data` directory and execute:
```
bash tinyimagenet_download.sh
```

### Training
All training scripts with different baselines and CURE framework can be found in the `scripts` folder, which consists of experiments on MNIST, CIFAR-10, and TinyImagenet. 

### Evaluation
One can use the onnx file from training outputs and verify the model using alpha-beta-crown. The config files can be found in `configs_abc` folder. 

#### Evaluation of Geometric Transformations
Please refer to the GitHub [https://github.com/uiuc-arc/CGT](https://github.com/uiuc-arc/CGT).

#### Evaluation of Patch Attacks
Please refer to the GitHub [https://github.com/Ping-C/certifiedpatchdefense](https://github.com/Ping-C/certifiedpatchdefense).


### Credits
Parts of the code in this repo is based on [https://github.com/eth-sri/SABR](https://github.com/eth-sri/SABR).

### Citation
Cite the paper/repo:
```
@article{jiang2024towards,
  title={Towards Universal Certified Robustness with Multi-Norm Training},
  author={Jiang, Enyi and Singh, Gagandeep},
  journal={arXiv preprint arXiv:2410.03000},
  year={2024}
}
```
