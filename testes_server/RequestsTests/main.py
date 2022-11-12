import numpy as np
import requests

clienteHttp = requests.session()

arr = np.random.randn(2, 2)

response = clienteHttp.post(
    url="http://127.0.0.1:8000/neuralserver/receive",
    data={'array0': arr, 'name': 'John', 'fal': [2, 4, 3], },
    # headers={"Content-Type": "multipart/form-data"}
)

response = clienteHttp.post(
    url="http://127.0.0.1:8000/neuralserver/receive",
    data={'array1': arr, 'name': 'John', 'fal': [2, 4, 3], },
    # headers={"Content-Type": "multipart/form-data"}
)

response = clienteHttp.get(
    url="http://127.0.0.1:8000/neuralserver/sinc",
    # headers={"Content-Type": "multipart/form-data"}
)

x = 1

response = clienteHttp.get(
    url="http://127.0.0.1:8000/neuralserver/sinc",
    # headers={"Content-Type": "multipart/form-data"}
)

x = 1
