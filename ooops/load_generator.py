import json
import time
from urllib import request
from urllib.request import urlopen

INPUTS = [
    {"features": [5.1, 3.5, 1.4, 0.2]},
    {"features": [3.1, 1.5, 2.4, 0.05]},
    {"features": [1.1, 1.2, 1.0, 0.02]},
]

def load_generator(url: str, interval: int = 1):
    i = 0
    while True:
        data = json.dumps(INPUTS[i % len(INPUTS)]).encode()
        req = request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")

        print(f"Sending request {i} with data {data}")
        out = urlopen(req)
        print(f"Response for request {i}: {out.read().decode()}")
        time.sleep(interval)
        i += 1

if __name__ == "__main__":
    load_generator("http://localhost:8000/predict")