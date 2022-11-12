from django.urls import path

from . import views

urlpatterns = [
    path('receive', views.receive_grads, name="receive"),
    path('sinc', views.sinc_grads, name="sinc"),
]
