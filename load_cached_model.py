from diffusers import DiffusionPipeline
import torch

prompt = "toy_face of a hacker with a hoodie"

pipe = DiffusionPipeline.from_pretrained("../pretrained", low_cpu_mem_usage=False).to("cuda")
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]

image.save("output2.png")
