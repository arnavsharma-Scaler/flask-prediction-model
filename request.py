import requests
import json

# Define the url of the predict_api endpoint
url = "http://localhost:5000/predict_api"

# Define the query data
data = {'query': 'your query text'}

# Send the POST request
response = requests.post(url, json=data)

# Print the status code and response text
print("Status code:", response.status_code)
print("Response text:", response.text)

# Try to parse the response as JSON
try:
	print(response.json())
except json.JSONDecodeError:
	print("Could not parse response as JSON")