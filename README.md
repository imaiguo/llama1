
# LLAMA1

Llama1 Content Length: 2K
Llama2 Content Length: 4K
Llama3 Content Length: 8K


## Windows环境部署

 准备独立的python环境

```bash
> cmd
> cd /opt/Data/PythonVenv
> python3 -m venv llama1
> source /opt/Data/PythonVenv/llama1/bin/activate
```

部署推理环境

```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 模型参数格式转换

bfb16转fb16 [pytorch训练格式 -> hugging face格式]

```bash
>
> python tools/convert_llama_weights_to_hf.py --input_dir /opt/Data/ModelWeight/meta/llama1/ --model_size 7B --output_dir /opt/Data/ModelWeight/meta/llama1.hf/llama1-7b-hf
> python tools/convert_llama_weights_to_hf.py --input_dir /opt/Data/ModelWeight/meta/llama1/ --model_size 13B --output_dir /opt/Data/ModelWeight/meta/llama1.hf/llama1-13b-hf
>
```

## 文本生成

```bash
> torchrun --nproc_per_node MP example.py --ckpt_dir /opt/Data/ModelWeight/meta/llama1/7B --tokenizer_path /opt/Data/ModelWeight/meta/llama1/tokenizer.model
>
> torchrun example.py --ckpt_dir /opt/Data/ModelWeight/meta/llama1/7B --tokenizer_path /opt/Data/ModelWeight/meta/llama1/tokenizer.model
```

启动服务
```bash
> python WebGradio/WebGradioAutoModelZiya.py
>
```

启动丘比特
```bash
> jupyter notebook --no-browser --port 7001 --ip=192.168.2.198
> jupyter notebook --no-browser --port 7000 --ip=192.168.2.200
```