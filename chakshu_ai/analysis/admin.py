from django.contrib import admin
from .models import Conversation, Message

@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('patient', 'start_time', 'title')
    list_filter = ('patient', 'start_time')
    search_fields = ('patient__name', 'title')

@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('conversation', 'message_type', 'timestamp')
    list_filter = ('message_type', 'timestamp', 'conversation__patient')
    search_fields = ('text', 'conversation__patient__name')
