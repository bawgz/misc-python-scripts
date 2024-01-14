import replicate

result = replicate.run(
        "bawgz/dripfusion-me:ac97cc0636ffea3721e59cec8acf774963e15ed2be67f3a51de3786e0b2b2946",
        input={
          "prompt": "a photo of tom felton man wearing reflective lens sunglasses, instagram",
          "negative_prompt": "((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), (fused fingers), (too many fingers), (((long neck)))",
          "refine": "expert_ensemble_refiner",
          "high_noise_frac": 0.95
        }
    )

print(result)

