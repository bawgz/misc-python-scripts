import replicate

training = replicate.trainings.create(
    version="bawgz/dripfusion-base:30f6d904d186c357901faf3429fecac93af142c449fbafe8bc5c2249ced608c0",
    input={
        "input_images": "https://replicate.delivery/pbxt/KAFzTB7svN2l7vJaiwbfL7rVzjSiijDGpCHDKhp8qBKSQjVR/me2.zip",
        "caption_prefix": 'A photo of TOK man, ',
        "use_face_detection_instead": True,
        "train_batch_size": 1,
        "max_train_steps": 4000,
        "lora_lr": 1e-4,
    },
    destination="bawgz/dripfusion-trained"
)

# https://replicate.delivery/pbxt/KAFzTB7svN2l7vJaiwbfL7rVzjSiijDGpCHDKhp8qBKSQjVR/me2.zip