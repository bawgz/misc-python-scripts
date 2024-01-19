import replicate

training = replicate.trainings.create(
    version="bawgz/dripfusion-base:614e1f80f6119ad02a601a9f5f8b1bffddff2f66fc42ad54f84158a6b53d7fdd",
    input={
        "input_images": "https://replicate.delivery/pbxt/KAFzTB7svN2l7vJaiwbfL7rVzjSiijDGpCHDKhp8qBKSQjVR/me2.zip",
        "caption_prefix": 'A photo of TOK man, ',
        # "use_face_detection_instead": True,
        "train_batch_size": 1,
        "max_train_steps": 4000,
        "lora_lr": 1e-4,
    },
    destination="bawgz/dripfusion-trained"
)

# https://replicate.delivery/pbxt/KAFzTB7svN2l7vJaiwbfL7rVzjSiijDGpCHDKhp8qBKSQjVR/me2.zip