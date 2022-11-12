import json
from multiprocessing import Queue

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

loss_dict = {}
max_loss_dict_size = 2

# A fila permite que os clientes esperem a sincronizacao sem busy wait
loss_queue = Queue()


@csrf_exempt
def receive_grads(request):
    global loss_dict, max_loss_dict_size, loss_queue

    loss_dict[list(request.POST.keys())[0]] = dict(request.POST)[list(request.POST.keys())[0]]

    # Colocamos uma copia do dicionario para cada cliente, se ele ja estiver completo
    if len(loss_dict.keys()) >= max_loss_dict_size:
        for _ in range(max_loss_dict_size):
            loss_queue.put(loss_dict)

        loss_dict = {}

    return HttpResponse()


def sinc_grads(request):
    x = 1
    return HttpResponse(json.dumps(loss_queue.get()))
