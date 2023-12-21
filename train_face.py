import replicate

training = replicate.trainings.create(
    version="bawgz/stable-dripfusion:cb7dc3fb600351875eb7ed4d2a4dc9842236aa2d19317cf33adaf954f0112f4e",
    input={
        "input_images": "https://replicate.delivery/pbxt/K5IafGgyGLIhAn3yqMSZe3H26Mrp44prmxorUpVhQnN7efQh/data.zip",
        "token_string": "LUK",
        "use_face_detection_instead": True,
    },
    destination="bawgz/stable-dripfusion"
)
