import requests

# Replace with your actual API Gateway URL
url = "https://<api-key>.execute-api.eu-west-2.amazonaws.com/prod/predict"
payload = {"texts": ["The waiting room was dirty, and the staff were rude."]}

headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
