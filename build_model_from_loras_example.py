from diffusers import DiffusionPipeline
import torch

pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
prompt = "toy_face of a hacker with a hoodie"

pipe = DiffusionPipeline.from_pretrained(pipe_id).to("cuda")

pipe.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy", low_cpu_mem_usage=False)

pipe.fuse_lora()
pipe.unload_lora_weights()

image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]

image.save("output.png")

pipe.save_pretrained("../pretrained")

pipe = DiffusionPipeline.from_pretrained("../pretrained").to("cuda")
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]

image.save("output2.png")

pipe.unfuse_lora()
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]

image.save("output3.png")
