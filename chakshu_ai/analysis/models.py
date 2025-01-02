from django.db import models
from patient.models import Patient

class Conversation(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='conversations')
    start_time = models.DateTimeField(auto_now_add=True)
    title = models.CharField(max_length=200, blank=True)  # Optional title for the conversation
    
    class Meta:
        ordering = ['-start_time']

    def __str__(self):
        return f"Conversation with {self.patient.name} at {self.start_time}"

class Message(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    timestamp = models.DateTimeField(auto_now_add=True)
    message_type = models.CharField(max_length=10, choices=[('user', 'User'), ('ai', 'AI')])
    text = models.TextField()
    image = models.TextField(null=True, blank=True)  # For base64 encoded images

    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f"{self.message_type} message in conversation {self.conversation.id}"
