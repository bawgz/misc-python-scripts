import torch
from diffusers import DiffusionPipeline

model = DiffusionPipeline.from_pretrained(
    "SG161222/RealVisXL_V3.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

model.load_lora_weights("bawgz/dripglasses_lora", weight_name="pit_viper_sunglasses.safetensors", adapter_name="SUN")
model.fuse_lora()
model.unload_lora_weights()

model.push_to_hub(repo_id="bawgz/dripfusion-base", variant="fp16")

pipe = DiffusionPipeline.from_pretrained(
  "bawgz/dripfusion-base",
  torch_dtype=torch.float16,
  use_safetensors=True,
  variant="fp16"
)

print("loaded model")

pipe = pipe.to("cuda")

print("to CUDAed model")

images = pipe("A photo of a man wearing pit viper sunglasses").images
# your output image
print(images[0])
images[0].save("output.png")
