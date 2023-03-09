from datetime import datetime

from django.db import models


class User(models.Model):
    username = models.CharField(max_length=200)
    password = models.IntegerField(max_length=200)


class Usertoken(models.Model):
    username = models.CharField(max_length=200)
    token = models.CharField(max_length=200)


def upload_to(instance,filename):
    return 'MedImgAI/SaveImage_temp/'+instance.dataFileID+filename

class Image(models.Model):
    projectID = models.CharField(max_length=200)
    dataFileID = models.CharField(max_length=200)
    addTime = models.DateTimeField(default=datetime.now())
    image = models.FileField(upload_to=upload_to)

class Database(models.Model):
    teamName = models.CharField(max_length=200)
    projectID = models.CharField(max_length=200)
    dataFileID = models.CharField(max_length=200)
    ifProcessed = models.CharField(max_length=200)


class Result(models.Model):
    projectID = models.CharField(max_length=200)
    dataFileID = models.CharField(max_length=200)
    brainInputImage = models.FileField(upload_to="MedImgAI/SaveImage_temp/")
    labelOutputImage = models.FileField(upload_to="MedImgAI/SaveImage_temp/")
    previewImage = models.FileField(upload_to="MedImgAI/SaveImage_temp/")
    labelOutput3DImage = models.FileField(upload_to="MedImgAI/SaveImage_temp/")


class Team(models.Model):
    teamName = models.CharField(max_length=200)
    owner = models.CharField(max_length=200)
    scale = models.PositiveIntegerField()
    isPublic = models.BooleanField(default=False)
    description = models.CharField(max_length=500)
    createdTime = models.DateTimeField(default=datetime.now())


class TeamInfo(models.Model):
    teamName = models.CharField(max_length=200)
    userName = models.CharField(max_length=200)
    status = models.CharField(max_length=200)
    createdTime = models.DateTimeField(default=datetime.now())


class Project(models.Model):
    projectID = models.CharField(max_length=200)
    teamName = models.CharField(max_length=200)
    isPublic = models.BooleanField(default=False)
    description = models.CharField(max_length=500)
    createdTime = models.DateTimeField(default=datetime.now())

class Train(models.Model):
    projectID = models.CharField(max_length=200)
    alpha = models.FloatField(max_length=200)
    epoch = models.PositiveIntegerField()
