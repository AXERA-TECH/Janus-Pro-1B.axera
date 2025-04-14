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
import argparse
import os


parser = argparse.ArgumentParser(description="Model configuration parameters")
parser.add_argument("--hf_model", type=str, default="Janus-Pro-1B",
                    help="Path to HuggingFace model")
parser.add_argument("--axmodel_path", type=str, default="janus_pro_1B_axmodel",
                    help="Path to save compiled axmodel of llama model")
parser.add_argument("-i", "--test_img_path", type=str, default="./imgs/image.png",
                    help="Test image path (supports png/jpg formats)")
parser.add_argument("--vit_axmodel_path", type=str, default="vit_axmodel/janus_warp_vit.axmodel",
                    help="Path to ViT model's axmodel")

args = parser.parse_args()

# base info
hf_model = args.hf_model
axmodel_path = args.axmodel_path
test_img_path = args.test_img_path
vit_axmodel_path = args.vit_axmodel_path
embeds = np.load(os.path.join(args.axmodel_path, "model.embed_tokens.weight.npy"))


def prepare_inputs_embeds(
    input_ids: torch.LongTensor,
    pixel_values: torch.FloatTensor,
    images_seq_mask: torch.LongTensor,
    images_emb_mask: torch.LongTensor,
    **kwargs,
):
    """

    Args:
        input_ids (torch.LongTensor): [b, T]
        pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
        images_seq_mask (torch.BoolTensor): [b, T]
        images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

        assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

    Returns:
        input_embeds (torch.Tensor): [b, T, D]
    """

    bs, n = pixel_values.shape[0:2]
    images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
    # [b x n, T2, D]
    vit_session = InferenceSession(vit_axmodel_path)
    images_embeds = vit_session.run(None, {"image": pixel_values[0].numpy()})[0] # pixel_values: [1, 1, 3, 384, 384]
    print(f"vit_output.shape is {images_embeds.shape}, vit feature extract done!")

    # [b x n, T2, D] -> [b, n x T2, D]
    images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
    # [b, n, T2] -> [b, n x T2]
    images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")

    # [b, T, D]
    input_ids[input_ids < 0] = 0  # ignore the image embeddings
    inputs_embeds = np.take(embeds, input_ids[0].cpu().numpy().tolist(), axis=0)[None, ...]
    inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

    return inputs_embeds

def post_process(data, topk=1, topp=0.9, temperature=0.6):
    def top_p(l: np.ndarray, p: float) -> np.ndarray:
        index = np.argsort(l)
        res = l.copy()
        sum_p = 0
        for i in index[::-1]:
            if sum_p >= p:
                res[i] = 0
            sum_p += res[i]
        return res / sum_p

    def softmax(l: np.ndarray) -> np.ndarray:
        l_max = l - l.max()
        l_exp = np.exp(l_max)
        res = l_exp / np.sum(l_exp)
        return res.astype(np.float64)

    r = data.astype(np.float32)
    r = r.flatten()
    candidate_index = np.argpartition(r, -topk)[-topk:]
    candidate_value = r[candidate_index]
    candidate_value /= temperature
    candidate_soft = softmax(candidate_value)
    candidate_soft = top_p(candidate_soft, topp)
    candidate_soft = candidate_soft.astype(np.float64) / candidate_soft.sum()
    pos = np.random.multinomial(1, candidate_soft).argmax()
    next_token = candidate_index[pos]
    return next_token, candidate_index, candidate_soft

config: MultiModalityConfig = AutoConfig.from_pretrained(hf_model, trust_remote_code=True)
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(hf_model)
tokenizer = vl_chat_processor.tokenizer

question = "请尝试理解这幅图中的内容."
conversation = [
    {
        "role": "User",
        "content": f"<image_placeholder>\n{question}",
        "images": [test_img_path],
    },
    {"role": "Assistant", "content": ""},
]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
)

input_embedding = prepare_inputs_embeds(**prepare_inputs)
token_ids = prepare_inputs['input_ids'].squeeze().numpy().tolist()
prefill_data = input_embedding
prefill_data = prefill_data.astype(bfloat16)
token_len = len(token_ids)

lastN = 1023
cfg = config.language_config

kv_dim = cfg.hidden_size // cfg.num_attention_heads * cfg.num_key_value_heads
k_caches = [
    np.zeros((1, lastN, kv_dim), dtype=bfloat16)
    for _ in range(cfg.num_hidden_layers)
]
v_caches = [
    np.zeros((1, lastN, kv_dim), dtype=bfloat16)
    for _ in range(cfg.num_hidden_layers)
]

prefill_decoder_sessins = []
for i in tqdm(range(cfg.num_hidden_layers), desc="Init InferenceSession"):
    session = InferenceSession(
        f"{axmodel_path}/llama_p640_l{i}_together.axmodel"
    )
    prefill_decoder_sessins.append(session)
post_process_session = InferenceSession(
    f"{axmodel_path}/llama_post.axmodel"
)
print("model load done!")

"""
    prefill
"""
prefill_len = 640

if prefill_len > 0:
    indices = np.array(list(range(prefill_len)), np.uint32).reshape(
        (1, prefill_len)
    )
    indices[:, token_len:] = 0
    mask = np.zeros((1, prefill_len, prefill_len)) - 65536
    data = np.zeros((1, prefill_len, cfg.hidden_size)).astype(bfloat16)
    data[:, 0:token_len] = prefill_data
    for i, t in enumerate(token_ids):
        mask[:, i, : i + 1] = 0
    mask = mask.astype(bfloat16)
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

post_out = post_process_session.run(None, {"input": data[:, token_len - 1, :][None, ...]})[0]
next_token, posssible_tokens, possible_soft = post_process(post_out, topk=1)
posibles = [tokenizer.decode([t]) for t in posssible_tokens]
posible_soft = [str((t, s)) for t, s in zip(posibles, possible_soft)]
token_ids.append(next_token)
print("prefill done!")

"""
    decode
"""
mask = np.zeros((1, 1, lastN + 1), dtype=np.float32).astype(bfloat16)
mask[:, :, :lastN] -= 65536
mask[:, :, :token_len] = 0
for start_indice in tqdm(range(lastN + 1), desc="Decoder"): # lastN + 1
    if prefill_len > 0 and start_indice < token_len:
        continue
    next_token = token_ids[start_indice]
    indices = np.array([start_indice], np.uint32).reshape((1, 1))
    data = embeds[next_token, :].reshape((1, 1, cfg.hidden_size)).astype(bfloat16)

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

    mask[..., start_indice] = 0
    if start_indice < token_len - 1:
        pass
    else:
        post_out = post_process_session.run(None, {"input": data})[0]
        next_token, posssible_tokens, possible_soft = post_process(post_out)
        token_ids.append(next_token)
    if next_token == tokenizer.eos_token_id:
        print("hit eos!")
        break
print("Janus Answers: ", tokenizer.decode(token_ids[token_len:], skip_special_tokens=True))
