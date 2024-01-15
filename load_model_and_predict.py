import torch
from diffusers import DiffusionPipeline

def main():
    pipe = DiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path="bawgz/dripfusion-base",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        token=True,
    )

    print("loaded model")

    pipe = pipe.to("cuda")

    images = pipe("A photo of a man wearing pit viper sunglasses").images
    # your output image
    images[0]



if __name__ == "__main__":
    main()