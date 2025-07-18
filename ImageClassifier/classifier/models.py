from django.db import models
from django.contrib.auth.models import User


class RecentAction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    model_name = models.CharField(max_length=50)
    predicted_label = models.CharField(max_length=50)
    confidence = models.FloatField() # accuracy

    def __str__(self):
        return f"{self.user.username} → {self.predicted_label} ({self.confidence:.2f})"
    
