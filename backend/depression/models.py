from django.db import models
from django.conf import settings
from django.contrib.auth.models import AbstractUser
from .managers import CustomUserManager

# Create your models here.

class User(AbstractUser):
	address = models.CharField(max_length=1000)

	objects = CustomUserManager()

	def __str__(self):
		return self.username


class DepressionnModel(models.Model):
    
    FacialVideo = models.FileField(null=True)
    Audio = models.FileField(null=True)
    
    UploadedDate = models.DateTimeField(auto_now_add=True)
    
    FacialOutput = models.FloatField(default=0.0)
    AudioOutput = models.FloatField(default=0.0)
    FinalResult = models.FloatField(default=0.0)
    
    DepressionStatus = models.CharField(max_length=255, default="Not depressed")
    
    UserID = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.DO_NOTHING)

    def __str__(self):
        return "Check " + str(self.id)