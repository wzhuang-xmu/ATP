# Installation  
Step 1: Create a new conda environment:
```
conda create -n atp python=3.10
conda activate atp
```
Step 2: Install relevant packages
```
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.47.0 datasets==3.4.1 numpy==1.23.0 wandb sentencepiece
pip install accelerate==0.26.0
```
