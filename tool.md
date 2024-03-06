## 建立虚拟环境 
    mkdir venv
    cd venv
    python -m venv venv_torch
    venv_torch\Scripts\activate
    pip list  # 查看包含的库

## vscode配置python内核

Ctrl+Shift+P
    
Python:select interpreter
    
选择虚拟环境中的python.exe位置（一般位于Scripts中）

## jupyter notebook版本问题会导致输出html文件受到影响

    pip install jupyter notebook==6.1.0

该版本可以正常运行

## Terminal debug python

    python -m pdb XXX.py

### command list

    l -- show current code
    n -- execute one line of code down
    s -- into this function
    p -- print a variable (p a, b)
    q -- quit debug
    ...

**Using the code below to set breakpoint** 
```python
import pdb; pdb.set_trace()
```

## Huggingface download(Linux)

### dependent
```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
```

### download model
```bash
huggingface-cli download --resume-download --repo-type model --local-dir-use-symlinks False bigscience/bloom-560m --local-dir bloom-560m
```

or below, when using below, u could not change the code in `.from_pretrained()` The dataset will be saved in ~/.cache/huggingface
```bash
huggingface-cli download --resume-download --repo-type model bigscience/bloom-560m
```

### download dataset
```
huggingface-cli download --resume-download --repo-type dataset --local-dir-use-symlinks False lavita/medical-qa-shared-task-v1-toy --local-dir data
```

or below. The dataset will be saved in ~/.cache/huggingface
```
huggingface-cli download --resume-download --repo-type dataset lavita/medical-qa-shared-task-v1-toy
```

### 使用export HF_ENDPOINT=https://hf-mirror.com 可以直接下载模型和数据

## Ubuntu使用screen保持离线后程序运行

创建screen

    screen -S XXX

查看screen

    screen -ls

进入screen

    screen -r XXX

## Optimize on Deep Learning

*使用Optuna进行超参数优化*

## Wandb使用注意事项

服务器上有时会使用不了wandb，显示wandb.init timeout什么的

    pip install urllib3==1.25.11

安装这个版本之后可以解决问题

## git上传

首先git clone + URL 下来文件

在clone里的文件里面进行修改

    $ git add .

    $ git commit -m "Add existing file"    

    $ git push origin YOUR_BRANCH(main)




