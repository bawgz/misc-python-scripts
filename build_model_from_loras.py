import torch
from diffusers import DiffusionPipeline

model = DiffusionPipeline.from_pretrained(
    "SG161222/RealVisXL_V3.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

# TODO: fuse a LoRA model into this model

model.load_lora_weights("bawgz/dripglasses_lora", weight_name="pit_viper_sunglasses.safetensors", adapter_name="SUN")

model.set_adapters("SUN", [1.0])

model.fuse_lora()

pipe = model.to("cuda")

print("loaded model")

images = pipe("A photo of a man wearing pit viper sunglasses").images
# your output image
images[0]

model.push_to_hub(repo_id="bawgz/dripfusion-base", token=True, private=True, variant="fp16", safe_serialization=True)
