import requests
import os

def upload_file_and_get_serving_url(api_token, file_path):
    # Initial POST request
    response = requests.post(
        f"https://dreambooth-api-experimental.replicate.com/v1/upload/{file_path}",
        headers={"Authorization": f"Token {api_token}"}
    )

    if response.status_code != 200:
        raise Exception(f"Failed to get URLs: {response.text}")

    # Extracting URLs from the response
    response_json = response.json()
    upload_url = response_json.get("upload_url")
    serving_url = response_json.get("serving_url")

    if not upload_url or not serving_url:
        raise Exception("Failed to extract URLs from the response")

    # Uploading the file
    with open(file_path, 'rb') as file:
        content_type = "application/zip" if file_path.endswith(".zip") else "application/x-tar"
        upload_response = requests.put(upload_url, data=file, headers={"Content-Type": content_type})

    if upload_response.status_code != 200:
        raise Exception(f"Failed to upload file: {upload_response.text}")

    return serving_url

# Replace 'your_api_token_here' with your actual API token and 'path_to_your_data.zip' with the path to your data.zip file
api_token = os.getenv('REPLICATE_API_TOKEN')
file_path = '1024_MAN.zip'

try:
    serving_url = upload_file_and_get_serving_url(api_token, file_path)
    print(f"Serving URL: {serving_url}")
except Exception as e:
    print(str(e))
