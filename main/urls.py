from django.urls import path
from .views import customer_service_query
from .views import customer_service
from .views import race_for_water_query

urlpatterns = [
    path('agent_query/', customer_service_query, name='agent_query'),
    path('agent_water/', race_for_water_query, name='agent_query'),
    path('agent_cyber/', customer_service, name='agent_query'),
]
