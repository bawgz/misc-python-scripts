import torch
from diffusers import DiffusionPipeline

model = DiffusionPipeline.from_pretrained(
    "SG161222/RealVisXL_V3.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

# TODO: fuse a LoRA model into this model

model.lora_lora_weights("bawgz/dripglasses_lora", weight_name="pit_viper_sunglasses-000003.safetensors", adapter_name="SUN")

model.set_adapters("SUN", [1.0])

model.fuse_lora()

model.push_to_hub("bawgz/dripfusion-base", token=True)