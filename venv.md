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
pip install -U hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

### download model
```bash
huggingface-cli download --resume-download --repo-type model --local-dir-use-symlinks False bigscience/bloom-560m --local-dir bloom-560m
```

### download dataset
```
huggingface-cli download --resume-download --repo-type dataset --local-dir-use-symlinks False dataset lavita/medical-qa-shared-task-v1-toy --local-dir data
```







