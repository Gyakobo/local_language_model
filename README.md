# Making a GPT (Generative Pretrained Transformer)
> In this project we'll basically try to model a character sequence of words. This work was mostly if not full inspired by the following publication [website](https://arxiv.org/pdf/1706.03762.pdf)

> It's most convenient if the following program is run using a GPU via conda Pytorch. I personally am running an EVGA 3070 Ti *(Needless to say, by the time you, my dear reader, are reading this this GPU might have been discontinued given that EVGA halted its GPU production)*.

> Note: Nvidia with the <span style="color:yellow">CUDA drivers</span> just don't work well with Linux, thus I'd be moving towards to AMD on my laptop for better results. So far also a dissapointment. 

> In the end what really helped me out was this command:
```bash
# Upgrade torch from 1.12 to 1.13
sudo pip3 install torch==1.13.0
```

## Installing and setting up Anaconda. 
Using <span style="color:yellow">wget</span> <span style="color:red">*(You might want to try it with 'sudo')*</span>:
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh 
chmod +x Anaconda3-2021.11-Linux-x86_64.sh
./Anaconda3-2021.11-Linux-x86_64.sh
```

Using <span style="color:yellow">curl</span> <span style="color:red">*(You might want to try it with 'sudo')*</span>:
```bash
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh 
chmod +x Anaconda3-2021.11-Linux-x86_64.sh
./Anaconda3-2021.11-Linux-x86_64.sh
```

Verify installation:
```bash
conda --version
```

Update Conda
```bash
conda update -n base -c defaults conda
```

## Creating a PyTorch virtual environment using Anaconda
1. **Create a New Environment**: Use the conda create command to create a new virtual environment. You can specify the Python version and any additional packages you need. For PyTorch, you can also specify the version of PyTorch you want to install. For example:

```bash
conda create -n myenv python=3.8
```
2. **Activate the Environment**: Once the environment is created, activate it using:

```bash
conda activate myenv 
```

> Note: You can delete a virtual environment with the following command:
```bash 
conda env list                      # Lists all the env(s) 
conda remove --name ENV_NAME --all  # Removes the select env
```

3. **Install PyTorch**: Once the environment is activated, you can install PyTorch using the conda install command. You should specify the appropriate version of PyTorch depending on your requirements. For example, to install PyTorch with CUDA support for GPU acceleration:

```bash
conda install pytorch torchvision torchaudio cudatoolkit -c pytorch 
```
4. **Verifying Cuda is working:**

> Note: Nvidia with the CUDA drivers just do not work well with Linux, thus I'd be moving towards to AMD on my laptop for better results

   1. After all that installations you need to cross-check whether CUDA is available. CUDA is a parallel computing platform and application programming interface (API) model created by Nvidia. In order for torch.cuda.is_available() to return True, you need to meet the following requirements:

      * **Nvidia GPU:** You need to have a Nvidia GPU installed on your system.
      * **CUDA Toolkit:** You need to have the CUDA Toolkit installed on your system. This includes CUDA drivers and the CUDA runtime library.
      * **PyTorch with CUDA support:** You need to have PyTorch installed with CUDA support. You can typically install the appropriate version of PyTorch using pip or conda, ensuring it matches the CUDA version installed on your system.
 
       > Here's a basic outline of the steps you might take to ensure torch.cuda.is_available() returns <span style=" color: green">**True**</span>:
 
      * **Check your GPU:** Ensure you have an Nvidia GPU installed on your system.
      * **Install CUDA Toolkit:** Download and install the CUDA Toolkit from Nvidia's website. Make sure to follow the installation instructions carefully.
      * **Install PyTorch with CUDA support:** Install PyTorch with CUDA support. If you're using pip, you might use a command like:
      ```bash
      pip install torch torchvision torchaudio
      ```
      * **Verify installation:** Once everything is installed, you can check if CUDA is available in Python using: 
      ```python
      import torch
      print(torch.cuda.is_available())
      ```

   2. In the case scenario that you don't have Cuda running please try running the following command:
   ```bash
   sudo apt-get install nvidia-cuda-toolkit
   ```
   > Note: To check the CUDA version run this command:
   ```bash
   nvcc --version
   ```
> Note to everyone programming and facing the password login issue with git:
```bash
$ git config credential.helper store
$ git push https://github.com/owner/repo.git

Username for 'https://github.com': <USERNAME>
Password for 'https://USERNAME@github.com': <PASSWORD>
```

> Note: Major update: This project shall be put on a Python Flask server and shared online
