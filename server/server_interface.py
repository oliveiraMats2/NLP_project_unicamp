import pickle
import urllib.parse

import requests


class ServerInterface(object):
    def __init__(self, model_name, server_adress="https://patrickctrf.loca.lt/", ):
        self.server_adress = server_adress
        self.model_name = model_name

        self.clienteHttp = requests.session()

    def share_loss(self, loss_tensor):
        return self.clienteHttp.post(
            url=urllib.parse.urljoin(self.server_adress, "neuralserver/receive_losses"),
            data=pickle.dumps(loss_tensor),
            headers={"model-name": self.model_name},
        )

    def receive_losses(self, ):
        response = self.clienteHttp.get(
            url=urllib.parse.urljoin(self.server_adress, "neuralserver/sinc_losses"),
        )

        # A resposta do servidor eh um dict serializado
        response_dict = pickle.loads(response.content)

        # Os values() do dicionario tb estao serializados. Entao desserializamos
        for key in response_dict.keys():
            response_dict[key] = pickle.loads(response_dict[key])

        return response_dict

    def share_weights(self, weights_tensor):
        return self.clienteHttp.post(
            url=urllib.parse.urljoin(self.server_adress, "neuralserver/receive_weights"),
            data=pickle.dumps(weights_tensor),
            headers={"model-name": self.model_name},
        )

    def receive_weights(self, ):
        response = self.clienteHttp.get(
            url=urllib.parse.urljoin(self.server_adress, "neuralserver/sinc_weights"),
        )

        # A resposta do servidor eh um dict serializado
        response_dict = pickle.loads(response.content)

        # Os values() do dicionario tb estao serializados. Entao desserializamos
        for key in response_dict.keys():
            response_dict[key] = pickle.loads(response_dict[key])

        return response_dict
