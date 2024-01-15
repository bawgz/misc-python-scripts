from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="dripglasses-pit_viper_sunglasses/model/pit_viper_sunglasses-000003.safetensors",
    path_in_repo="pit_viper_sunglasses-000003.safetensors",
    repo_id="bawgz/dripglasses_lora",
    repo_type="model",
)