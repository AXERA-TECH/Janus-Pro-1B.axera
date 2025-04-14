# REF: https://github.com/deepseek-ai/Janus
import numpy as np
import torch
from axengine import InferenceSession
from ml_dtypes import bfloat16
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from tqdm import tqdm
from einops import rearrange
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.models.modeling_vlm import MultiModalityConfig
from janus.utils.io import load_pil_images
import os
import PIL.Image
from loguru import logger
import onnxruntime
import argparse


parser = argparse.ArgumentParser(description="Model configuration parameters")
parser.add_argument("--hf_model", type=str, default="Janus-Pro-1B",
                    help="Path to HuggingFace model")
parser.add_argument("--axmodel_path", type=str, default="janus_pro_1B_axmodel",
                    help="Path to save compiled axmodel of llama model")
args = parser.parse_args()


# base info
hf_model = args.hf_model
axmodel_path = args.axmodel_path

"""ONNX MODEL"""
gen_vision_model_decode = onnxruntime.InferenceSession("./img_gen_onnx/gen_vision_model_decode_sim.onnx", providers=["CPUExecutionProvider"])
gen_aligner = onnxruntime.InferenceSession("./img_gen_onnx/gen_aligner.onnx", providers=["CPUExecutionProvider"])
gen_head = onnxruntime.InferenceSession("./img_gen_onnx/post_head.onnx", providers=["CPUExecutionProvider"])
post_norm = onnxruntime.InferenceSession("./img_gen_onnx/post_norm.onnx", providers=["CPUExecutionProvider"])
"""ONNX MODEL"""

"""EMBEDINGs"""
embeds = np.load(f"{axmodel_path}/model.embed_tokens.weight.npy")
gen_embed = np.load("./embeds/gen_embed.npy")
codebook_entry_embedding = torch.load('./embeds/codebook_entry_embedding.pt', map_location=torch.device('cpu'))
"""EMBEDINGs"""


def prefill(
    cfg,
    prefill_decoder_sessins,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 1,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int)
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1: -1] = vl_chat_processor.pad_id

    inputs_embeds = embeds[tokens.numpy()]
    batch, token_len, seq_dim = inputs_embeds.shape
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int)
    prefill_len = 640
    token_ids = tokens

    ###################################################################
    lastN = 1023
    kv_dim = cfg.hidden_size // cfg.num_attention_heads * cfg.num_key_value_heads
    batch_k_caches = {}
    batch_v_caches = {}

    for bid in range(batch):
        batch_k_caches[bid] = [
            np.zeros((1, lastN, kv_dim), dtype=bfloat16)
            for _ in range(cfg.num_hidden_layers)
        ]
        batch_v_caches[bid] = [
            np.zeros((1, lastN, kv_dim), dtype=bfloat16)
            for _ in range(cfg.num_hidden_layers)
        ]
    ###################################################################
    mask = np.zeros((1, prefill_len, prefill_len)) - 65536
    for j in range(token_len):
        mask[:, j, :j + 1] = 0
    mask = mask.astype(bfloat16)

    indices = np.array(list(range(prefill_len)), np.uint32).reshape(
        (1, prefill_len)
    )
    indices[:, token_len:] = 0
    hidden_states = np.zeros((batch, token_len, cfg.hidden_size)).astype(bfloat16)

    for bid in range(batch):
        data = np.zeros((1, prefill_len, cfg.hidden_size)).astype(bfloat16)
        data[:, 0:token_len] = inputs_embeds[bid].astype(bfloat16)
        k_caches = batch_k_caches[bid]
        v_caches = batch_v_caches[bid]

        for i in range(cfg.num_hidden_layers):
            input_feed = {
                "K_cache": np.zeros((1, 1, cfg.hidden_size), dtype=bfloat16),
                "V_cache": np.zeros((1, 1, cfg.hidden_size), dtype=bfloat16),
                "indices": indices,
                "input": data,
                "mask": mask,
            }
            outputs = prefill_decoder_sessins[i].run(None, input_feed, shape_group=1)
            k_caches[i][:, :token_len, :] = outputs[0][:, :token_len, :]
            v_caches[i][:, :token_len, :] = outputs[1][:, :token_len, :]
            data[:, :token_len] = outputs[2][:, :token_len, :]

        ######## BATCH ###########
        hidden_states[bid] = data[:, :token_len]
        batch_k_caches[bid] = k_caches
        batch_v_caches[bid] = v_caches

    ################# NORM & GEN-HEAD ########################
    hidden_states = post_norm.run(["output"], {"input": hidden_states[:, -1:, :].astype(np.float32)})[0]
    logits = gen_head.run(["output"], {"input": hidden_states[:, -1, :]})[0] # 与 llama head 不同, 此 head 为图像生成专用
    ############# POST & GET NEXT TOKEN #############
    logits = torch.from_numpy(logits)
    logit_cond = logits[0::2, :]
    logit_uncond = logits[1::2, :]
    logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
    probs = torch.softmax(logits / temperature, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    generated_tokens[:, 0] = next_token.squeeze(dim=-1)
    next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
    ################## PREPARE_GEN_IMG_EMBEDS ##################
    gen_embed_res = np.take(gen_embed, next_token.numpy().tolist(), axis=0)
    img_embeds = gen_aligner.run(["output"], {"input": gen_embed_res})[0]
    inputs_embeds = np.expand_dims(img_embeds, axis=1)
    return inputs_embeds, token_ids, generated_tokens, batch_k_caches, batch_v_caches


@torch.inference_mode()
def generate(
    cfg,
    prefill_decoder_sessins,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 1, # 目前只支持固定为 1
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    inputs_embeds, token_ids, generated_tokens, batch_k_caches, batch_v_caches = prefill(
        cfg, prefill_decoder_sessins, vl_chat_processor,
        prompt, temperature, parallel_size, cfg_weight, image_token_num_per_image
    )

    logger.debug("prefill completed!")
    token_len = token_ids.shape[1]

    lastN = 1023

    batch = parallel_size * 2

    mask = np.zeros((1, 1, lastN + 1), dtype=np.float32).astype(bfloat16)
    mask[:, :, :lastN] -= 65536
    mask[:, :, :token_len] = 0

    for image_token_i in tqdm(range(1, image_token_num_per_image), desc="ImageToken"):

        # 下面是 decode 逻辑
        start_indice = image_token_i + token_len - 1
        indices = np.array([start_indice], np.uint32).reshape((1, 1))
        hidden_states = np.zeros((batch, 1, cfg.hidden_size)).astype(bfloat16) # batch, 1, seq_dim
        assert (inputs_embeds[0] == inputs_embeds[1]).all()

        for bid in range(batch):
            k_caches = batch_k_caches[bid]
            v_caches = batch_v_caches[bid]
            data = inputs_embeds[:1, ...].astype(bfloat16)

            for i in range(cfg.num_hidden_layers):
                input_feed = {
                    "K_cache": k_caches[i],
                    "V_cache": v_caches[i],
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }

                outputs = prefill_decoder_sessins[i].run(None, input_feed, shape_group=0)
                k_caches[i][:, start_indice, :] = outputs[0][:, :, :]
                v_caches[i][:, start_indice, :] = outputs[1][:, :, :]
                data = outputs[2]

            hidden_states[bid] = data
            batch_k_caches[bid] = k_caches
            batch_v_caches[bid] = v_caches

        mask[..., start_indice] = 0

        ############### NORM & GEN_HEAD #######################
        hidden_states = post_norm.run(["output"], {"input": hidden_states.astype(np.float32)})[0]
        logits = gen_head.run(["output"], {"input": hidden_states[:, -1, :]})[0]
        ############# POST & GET NEXT TOKEN #############
        logits = torch.from_numpy(logits)
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, image_token_i] = next_token.squeeze(dim=-1)
        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        ################## PREPARE_GEN_IMG_EMBEDS ##################
        gen_embed_res = np.take(gen_embed, next_token.numpy().tolist(), axis=0)
        img_embeds = gen_aligner.run(["output"], {"input": gen_embed_res})[0]
        inputs_embeds = np.expand_dims(img_embeds, axis=1)

    # z_q 为 vision decode 的输出
    indices = generated_tokens.to(dtype=torch.int)
    shape = [parallel_size, 8, img_size//patch_size, img_size//patch_size]
    z_q = codebook_entry_embedding[indices]  # (b*h*w, c)
    z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
    # reshape back to match original input shape
    z_q = z_q.permute(0, 3, 1, 2)
    dec = gen_vision_model_decode.run(['image'], {'quant': z_q.to(dtype=torch.float32).numpy()})[0]
    dec = dec.transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)
    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs('generated_samples', exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join('generated_samples', "img_{}.jpg".format(i))
        PIL.Image.fromarray(visual_img[i]).save(save_path)

###################################################################
config: MultiModalityConfig = AutoConfig.from_pretrained(hf_model, trust_remote_code=True)
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(hf_model)
tokenizer = vl_chat_processor.tokenizer

description = "A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue."

conversation = [
    {
        "role": "User",
        "content": description,
    },
    {"role": "Assistant", "content": ""},
]

sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=vl_chat_processor.sft_format,
    system_prompt="",
)
prompt = sft_format + vl_chat_processor.image_start_tag
###################################################################

cfg = config.language_config

prefill_decoder_sessins = []
for i in tqdm(range(cfg.num_hidden_layers), desc="Init InferenceSession"):
    session = InferenceSession(
        f"{axmodel_path}/llama_p640_l{i}_together.axmodel"
    )
    prefill_decoder_sessins.append(session)

logger.info("model load done!")

generate(
    cfg,
    prefill_decoder_sessins,
    vl_chat_processor,
    prompt
)
