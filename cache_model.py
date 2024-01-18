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

print("state dict before fuse", state_dict_before_fuse)

pipe.fuse_lora()

print("pipe fused")
print(pipe.unet)

state_dict_after_fuse = pipe.unet.state_dict().copy()

print("state dict after fuse", state_dict_before_fuse)

print("state dict before fuse == state dict after fuse", state_dict_before_fuse == state_dict_after_fuse)

prompt = "toy_face of a hacker with a hoodie"
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]

image.save("output.png")

pipe.save_pretrained("../pretrained")

print("pipe saved")
print(pipe.unet)
