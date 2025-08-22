import requests

url = "https://<api-key>.execute-api.eu-west-2.amazonaws.com/prod/predict"

payload = {"review": "The staff was extremely kind and helpful!"}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
