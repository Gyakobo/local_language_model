# Making a GPT (Generative Pretrained Transformer)
> In this project we'll basically try to model a character sequence of words. This work was mostly if not full inspired by the following publication [website](https://arxiv.org/pdf/1706.03762.pdf)

> It's most convenient if the following program is run using a GPU via conda Pytorch

## Installing and setting up Anaconda
Using wget:
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
chmod +x Anaconda3-2021.11-Linux-x86_64.sh
./Anaconda3-2021.11-Linux-x86_64.sh
```

Using curl:
```bash
curl -O https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
chmod +x Anaconda3-2021.11-Linux-x86_64.sh
./Anaconda3-2021.11-Linux-x86_64.sh
```

Verify installation:
```bash
conda --version
```

## Creating a PyTorch virtual environment using Anaconda
1. Create a New Environment: Use the conda create command to create a new virtual environment. You can specify the Python version and any additional packages you need. For PyTorch, you can also specify the version of PyTorch you want to install. For example:

```bash
conda create -n myenv python=3.8
```
2. Activate the Environment: Once the environment is created, activate it using:

```bash
conda activate myenv 
```

> Note: You can delete a virtual environment with the following command:
```bash 
conda env list                      # Lists all the env(s) 
conda remove --name ENV_NAME --all  # Removes the select env
```




