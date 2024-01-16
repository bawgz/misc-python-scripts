from diffusers import DiffusionPipeline
import torch

pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16, variant="fp16", low_cpu_mem_usage=False).to("cuda")

pipe.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy", low_cpu_mem_usage=False)

pipe.fuse_lora()

prompt = "toy_face of a hacker with a hoodie"
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]

image.save("output.png")

pipe.save_pretrained("../pretrained", variant="fp16", safe_serialization=True)

pipe = DiffusionPipeline.from_pretrained("../pretrained", torch_dtype=torch.float16, variant="fp16", low_cpu_mem_usage=False).to("cuda")
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]

print("?????????????????")

image.save("output2.png")
