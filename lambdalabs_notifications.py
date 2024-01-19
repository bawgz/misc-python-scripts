import os
import time
import requests
from discord import SyncWebhook, Embed

while True:
  response = requests.get(
    "https://cloud.lambdalabs.com/api/v1/instance-types",
    headers={ "cookie": os.getenv('LAMBDALABS_COOKIE') }
  )

  if response.status_code != 200:
    raise Exception(f"Failed to get fetch GPUs: {response.text}")

  response_json = response.json()['data']

  print(response_json)
  print("____________________________________________________________________")

  filtered_dict = {k:v for (k,v) in response_json.items() if v['instance_type']['price_cents_per_hour'] < 100 and len(v['regions_with_capacity_available']) > 0}

  print(filtered_dict)

  if len(filtered_dict) > 0:
    description = "\n".join([f"{v['instance_type']['description']}: ${v['instance_type']['price_cents_per_hour'] / 100} / hr" for k,v in filtered_dict.items()])

    embeds = [Embed(title="GPU available! :rocket:", description=description, color=0x00ff00, url="https://cloud.lambdalabs.com/instances")]

    webhook = SyncWebhook.from_url(os.getenv('DISCORD_WEBHOOK_LAMBDA_LABS'))
    webhook.send(embeds=embeds)
    time.sleep(300)

  time.sleep(30)
