# Janus-Pro-1B.axera

> Deepseek Janus-Pro-1B DEMO on Axera.

- 目前支持 `Python` 语言, `C++` 代码在开发中
- 预编译模型下载 [Google Drive](https://drive.google.com/drive/folders/1ZX_tOjqrVIRWArcalbxjzwrtE9kWzctr?usp=sharing) 或 [百度网盘](https://pan.baidu.com/s/1Ht-q22Lj3KwY_TT-K27lCw?pwd=jhdk)下载
- 如需自行转换 VIT 模型请参考 [模型转换](/model_convert/README.md)

## 支持平台

- [x] AX650N
- [ ] AX630C

## Git Clone

首先使用如下命令 `clone` 本项目, 然后进入 `python` 文件夹:

```bash
$ git clone git@github.com:AXERA-TECH/Janus-Pro-1B.axera.git
$ cd Janus-Pro-1B.axera/python
```

之后在开发板上下载或安装以下支持库:

- 安装 `janus` 支持, 确保 `from janus.models import MultiModalityCausalLM, VLChatProcessor` 导入不报错

    ```bash
    $ git clone https://github.com/deepseek-ai/Janus
    $ cd Janus
    $ pip3 install -e .
    ```

- 从 `huggingface` 下载 `Janus-Pro-1B` 模型.

    ```bash
    $ git clone https://huggingface.co/deepseek-ai/Janus-Pro-1B
    ```

- 在开发板上安装配置 `pyaxengine`, [点击跳转下载链接](https://github.com/AXERA-TECH/pyaxengine/releases). 注意板端 `SDK` 最低版本要求:
    
    - AX650 SDK >= 2.18
    - AX620E SDK >= 3.12
    - 执行 `pip3 install axengine-x.x.x-py3-none-any.whl` 安装

将下载后的预编译模型解压到当前文件夹 (*可选, 之后可以手动指定模型路径*), 默认文件夹排布如下:

```bash
.
├── embeds
│   ├── codebook_entry_embedding.npy
│   ├── codebook_entry_embedding.pt
│   └── gen_embed.npy
├── generated_samples
├── img_gen_onnx
│   ├── gen_aligner.onnx
│   ├── gen_vision_model_decode_sim.onnx
│   ├── post_head.onnx
│   └── post_norm.onnx
├── imgs
│   ├── image.jpg
│   └── image.png
├── infer_axmodel_gen.py
├── infer_axmodel_und.py
├── Janus-Pro-1B
│   ├── config.json
│   ├── janus_pro_teaser1.png
│   ├── janus_pro_teaser2.png
│   ├── preprocessor_config.json
│   ├── processor_config.json
│   ├── README.md
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── tokenizer.json
├── janus_pro_1B_axmodel
│   ├── llama_p640_l0_together.axmodel
│   ├── llama_p640_l10_together.axmodel
│   ├── llama_p640_l11_together.axmodel
│   ├── llama_p640_l12_together.axmodel
│   ├── llama_p640_l13_together.axmodel
│   ├── llama_p640_l14_together.axmodel
│   ├── llama_p640_l15_together.axmodel
│   ├── llama_p640_l16_together.axmodel
│   ├── llama_p640_l17_together.axmodel
│   ├── llama_p640_l18_together.axmodel
│   ├── llama_p640_l19_together.axmodel
│   ├── llama_p640_l1_together.axmodel
│   ├── llama_p640_l20_together.axmodel
│   ├── llama_p640_l21_together.axmodel
│   ├── llama_p640_l22_together.axmodel
│   ├── llama_p640_l23_together.axmodel
│   ├── llama_p640_l2_together.axmodel
│   ├── llama_p640_l3_together.axmodel
│   ├── llama_p640_l4_together.axmodel
│   ├── llama_p640_l5_together.axmodel
│   ├── llama_p640_l6_together.axmodel
│   ├── llama_p640_l7_together.axmodel
│   ├── llama_p640_l8_together.axmodel
│   ├── llama_p640_l9_together.axmodel
│   ├── llama_post.axmodel
│   └── model.embed_tokens.weight.npy
├── requirements.txt
└── vit_axmodel
    └── janus_warp_vit.axmodel
```

## 模型转换

关于 `onnx` 和 `axmodel` 的导出、编译参见 [模型转换](./model_convert/README.md) 部分内容.

## 上板部署

- `AX650N` 的设备已预装 `Ubuntu 22.04`
- 以 `root` 权限登陆 `AX650N` 的板卡设备
- 接入互联网, 确保 `AX650N` 的设备能正常执行 `apt install`, `pip install` 等指令
- 已验证设备: `AX650N DEMO Board`、`爱芯派Pro(AX650N)`

### Python API 运行

#### Requirements

```bash
$ mkdir /opt/site-packages
$ cd python
$ pip3 install -r requirements.txt --prefix=/opt/site-packages
``` 

#### 添加环境变量

将以下两行添加到 `/root/.bashrc`(实际添加的路径需要自行检查)后, 重新连接终端或者执行 `source ~/.bashrc`

```bash
$ export PYTHONPATH=$PYTHONPATH:/opt/site-packages/local/lib/python3.10/dist-packages  
$ export PATH=$PATH:/opt/site-packages/local/bin
``` 

#### 运行

在 `Axera 开发板` 上运行以下命令开启图像理解功能:

```sh
$ cd Janus-Pro-1B.axera/python
$ python3 infer_axmodel_und.py
```

默认输入图像为:

![image.png](python/imgs/image.png)

也可以通过命令行参数手动指定图像路径. 模型推理结果如下:

```bash
[INFO] Chip type: ChipType.MC50
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Engine version: 2.11.0a
vit_output.shape is (1, 576, 2048), vit feature extract done!
Init InferenceSession: 100%|██████████████████████████████████████████████████████████| 24/24 [00:06<00:00,  3.89it/s]
model load done!
prefill done!
Decoder:  62%|█████████████████████████████████████████▍                         | 634/1024 [00:00<00:00, 2493.31it/s]Decoder:  72%|█████████████████████████████████████████████████▍                   | 733/1024 [00:18<00:09, 29.61it/s]hit eos!
Decoder:  74%|███████████████████████████████████████████████████▎                 | 762/1024 [00:23<00:08, 32.02it/s]
这幅图展示了三位穿着宇航服的宇航员，他们站在一片茂密的植被中。宇航员们的头盔上有反光面罩，可以看到他们的面容。背景是一片森林，树木和植物的细节非常清晰。宇航员们的姿势各不相同，其中一位宇航员正举起双手，似乎在向某人挥手，另一位宇航员则站立着，目光向前方看去，第三位宇航员则弯腰靠近地面，似乎在观察地面上的某物。整体画面给人一种科幻和探索的感觉，仿佛他们正在进行一次太空探险任务。
```

在 `Axera 开发板` 上运行以下命令实现图像生成:

```sh
$ cd Janus-Pro-1B.axera/python
$ python3 infer_axmodel_gen.py
```

默认 prompt: `"A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue."`

生成的图像默认保存在 `./generated_samples/` 文件夹下:

![output](assets/gen_out_img.jpg)

#### 图像理解任务·推理耗时统计

Model | Time |
---| ---|
ImageEncoder | 142.682 ms |
Prefill | 4560.214 ms |
Decoder | 87.48 ms |

其中:

- `Prefill` 阶段, 每一层的 `llama_layer` 平均耗时 `189.565 ms`.
- `Decoder` 阶段, 每一层的 `llama_layer` 平均耗时 `3.201` ms.
- `llama_post` 耗时 `10.654 ms`.

模型解码速度为: 1000 / 87.48 = 11.43 token/s.

#### 图像生成任务·推理耗时统计 (one token)

Model | Time |
---| ---|
llama prefill g1 | 189.565 ms * 24 * 2 |
llama decode g0 | 3.201 ms * 24 * 2 |
norm & gen_head | 40 ms
gen_aligner | 2.0 ms

最后使用 `gen_vision_model_decode` 获取图像结果, 该模块耗时 `17507.68 ms`.

## 技术讨论

- Github issues
- QQ 群: 139953715
