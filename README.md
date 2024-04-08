# Making a GPT (Generative Pretrained Transformer)
> In this project we'll basically try to model a character sequence of words. This work was mostly if not full inspired by the following publication [website](https://arxiv.org/pdf/1706.03762.pdf)

> It's most convenient if the following program is run using a GPU via conda Pytorch

### Installing and setting up Anaconda
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
```shell script
conda --version
```


