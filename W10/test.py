import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
#url = 'https://yrhjb2i5yl.execute-api.us-east-1.amazonaws.com/test/predict'

#data = {'url':'http://bit.ly/mlbookcamp-pants'}
data = {'url':'https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg'}
result = requests.post(url, json=data).json()

print(result)
