from django.contrib import admin
from django.urls import path
from research.views import index, chat_api  # use absolute import

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name='index'),
    path('api/chat/', chat_api, name='chat_api'),
]
