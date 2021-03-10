import requests

response = requests.post(url="http://0.0.0.0:5000", data="test.mp4")
print(response.text)