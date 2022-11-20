import pickle

import numpy as np
import requests

clienteHttp = requests.session()

arr = np.random.randn(2, 2)

response = clienteHttp.post(
    url="https://patrickctrf.loca.lt/neuralserver/receive_losses",
    data=pickle.dumps(arr),
    headers={"model-name": "alfa"},
    # headers={"Content-Type": "multipart/form-data"}
)

response = clienteHttp.post(
    url="https://patrickctrf.loca.lt/neuralserver/receive_losses",
    data=pickle.dumps(arr),
    headers={"model-name": "omega"},
    # headers={"Content-Type": "multipart/form-data"}
)

response = clienteHttp.get(
    url="https://patrickctrf.loca.lt/neuralserver/sinc_losses",
    # headers={"Content-Type": "multipart/form-data"}
)

x = 1

response = clienteHttp.get(
    url="https://patrickctrf.loca.lt/neuralserver/sinc_losses",
    # headers={"Content-Type": "multipart/form-data"}
)

x = 1

# response = clienteHttp.post(
#     url="https://patrickctrf.loca.lt/neuralserver/receive",
#     data={'model0': str(arr), 'name': 'John', 'fal': [2, 4, 3], },
#     headers={"model-name": "alfa"},
#     # headers={"Content-Type": "multipart/form-data"}
# )