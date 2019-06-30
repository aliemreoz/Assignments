from __future__ import unicode_literals

from django.contrib.auth.models import User
from django.db import models
from tags.models import Tag
# Create your models here.

class Blog(models.Model):

    director = models.CharField(max_length=220)
    movie = models.CharField(max_length=520)
    owner = models.ForeignKey(User)
    tags = models.ManyToManyField(Tag)