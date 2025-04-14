import torch
import argparse
from loguru import logger
import os
import onnx
from onnx.shape_inference import infer_shapes
from onnxsim import simplify
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM
import torch.nn as nn


def onnx_sim(onnx_path):
    onnx_model = onnx.load(onnx_path)
    onnx_model = infer_shapes(onnx_model)
    # convert model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_path)
    logger.info(f"onnx simpilfy successed, and model saved in {onnx_path}")


if __name__ == '__main__':
    
    """
    Usage:
        python3 export_onnx.py -m /path/your/hugging_face/models/Janus-Pro-1B/ -o ./vit-models
    """
    parser = argparse.ArgumentParser(prog='main')
    parser.add_argument("-m", "--model", type=str, help="hugging fance model path")
    parser.add_argument("-o", "--onnx_save_dir", type=str, default='./vit-models', help="vit onnx model save path")
    args = parser.parse_args()

    model_path = args.model
    onnx_save_dir = args.onnx_save_dir

    if not os.path.exists(onnx_save_dir):
        os.makedirs(onnx_save_dir)

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )

    janus_warp_model_onnx_save_dir = os.path.join(
        onnx_save_dir,
        'janus_warp_vit_model.onnx'
    )

    images = torch.randn(1, 3, 384, 384).to(vl_gpt.device)

    class WarpVisionModel(nn.Module):
        def __init__(self, vision_model, aligner):
            super().__init__()
            self.vision_model = vision_model
            self.aligner = aligner

        def forward(self, images):
            return self.aligner(self.vision_model(images))

    warp_model = WarpVisionModel(vl_gpt.vision_model, vl_gpt.aligner)
    torch.onnx.export(
        warp_model.to(dtype=torch.float32),
        images.to(dtype=torch.float32),
        janus_warp_model_onnx_save_dir,
        opset_version=17, # 14
        do_constant_folding=True,
        verbose=False,
        input_names=["image"],
        output_names=["output"],
    )
    logger.debug("export janus_warp_model onnx succee!")
    onnx_sim(janus_warp_model_onnx_save_dir)
