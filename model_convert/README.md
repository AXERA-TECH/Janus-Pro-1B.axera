# 模型转换

## 环境配置

创建虚拟环境

```bash
$ conda create -n Janus-Pro-1B python=3.11 -y
$ conda activate Janus-Pro-1B
```

常规依赖安装:

```bash
$ pip3 install -r requirements_v2.txt
```

从源码安装 `janus` 支持, 确保 `from janus.models import MultiModalityCausalLM, VLChatProcessor` 导入不报错.

```bash
$ git clone https://github.com/deepseek-ai/Janus
$ cd Janus
$ pip3 install -e .
```

## 导出 Vit-ONNX 模型 (PyTorch -> ONNX)

示例命令如下:

```bash
$ python3 export_onnx.py -m /path/your/hugging_face/models/Janus-Pro-1B/ -o ./vit-models
```

其中 `-m` 参数需要指定 `hugging_face Janus-Pro-1B` 模型路径, 如果模型不存在, 可以通过以下命令下载:

```bash
$ git clone https://huggingface.co/deepseek-ai/Janus-Pro-1B
```

模型成功导出成功后会在 `vit-models` 目录中生成所需要的 `onnx` 模型.

## 模型编译 (ONNX -> AXmodel)

使用模型转换工具 `Pulsar2` 将 `ONNX` 模型转换成适用于 `Axera-NPU` 运行的模型文件格式 `.axmodel`, 通常情况下需要经过以下两个步骤:

- 生成适用于该模型的 `PTQ` 量化校准数据集
- 使用 `Pulsar2 build` 命令集进行模型转换（PTQ 量化、编译）, 更详细的使用说明请参考 [AXera Pulsar2 工具链指导手册](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)

### 下载量化数据集

```sh
$ bash download_dataset.sh
```

执行结束后可以在当前文件夹内看到名为 `imagenet-calib.tar` 的量化文件.

### 修改配置文件
 
在 `pulsar2_configs` 目录中, 检查 `*.json` 中 `calibration_dataset` 字段, 将该字段配置的路径改为上一步下载的量化数据集存放路径, 通常可以是 `.tar` 或 `.zip` 文件.

### Pulsar2 build 编译

示例命令如下:

```bash
$ pulsar2 build --output_dir ./compiled_output_janus_warp_model --config config/janus_vit_config.json --npu_mode NPU3 --input model/janus_warp_vit_model.onnx --target_hardware AX650
```

关于 `pulsar2 build` 更详细的文档请参考 [Pulsar2-QuickStart](https://npu.pages-git-ext.axera-tech.com/pulsar2-docs/user_guides_quick/quick_start_ax650.html).
