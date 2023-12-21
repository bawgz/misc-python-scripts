import replicate

training = replicate.trainings.create(
    version="stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
    input={
        "input_images": "https://replicate.delivery/pbxt/K5GU1D1u6jwHw8wVni1MLAtuH1BaPtdDDztVWj4nqVw3Vf0Y/data.zip",
        "lora_lr": 2e-4,
        "caption_prefix": 'A photo of TOK sunglasses,',
    },
    destination="bawgz/stable-dripfusion"
)
