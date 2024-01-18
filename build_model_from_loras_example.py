from diffusers import DiffusionPipeline
import torch

pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
prompt = "toy_face of a hacker with a hoodie"

pipe = DiffusionPipeline.from_pretrained(pipe_id).to("cuda")

unet_keys_at_create = pipe.unet.state_dict().keys()

print("unet keys at create", unet_keys_at_create)


pipe.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy", low_cpu_mem_usage=False)

unet_keys_at_load_lora = pipe.unet.state_dict().keys()

print("unet keys at load lora", unet_keys_at_load_lora)

pipe.fuse_lora()

image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]

image.save("output.png")

unet_keys_at_save = pipe.unet.state_dict().keys()

print("unet keys at save", unet_keys_at_save)

pipe.save_pretrained("../pretrained")

pipe = DiffusionPipeline.from_pretrained("../pretrained").to("cuda")
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]

image.save("output2.png")
