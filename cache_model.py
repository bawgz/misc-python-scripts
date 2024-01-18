from diffusers import DiffusionPipeline
import torch

pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(pipe_id).to("cuda")

print("pipe created")
print(pipe.unet)

pipe.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy")

print("pipe LoRA loaded")
print(pipe.unet)

state_dict_before_fuse = pipe.unet.state_dict().copy()

pipe.fuse_lora()

print("pipe fused")
print(pipe.unet)

state_dict_after_fuse = pipe.unet.state_dict().copy()

def dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    shared_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o : (d1[o], d2[o]) for o in shared_keys if not torch.eq(d1[o], d2[o])}
    return added, removed, modified

added, removed, modified = dict_compare(state_dict_before_fuse, state_dict_after_fuse)

print("added:", added)
print("removed:", removed)
print("modified:", modified)

prompt = "toy_face of a hacker with a hoodie"
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]

image.save("output.png")

pipe.save_pretrained("../pretrained")

print("pipe saved")
print(pipe.unet)
