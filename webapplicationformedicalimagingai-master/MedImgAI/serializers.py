from rest_framework import serializers
from rest_framework.exceptions import ValidationError

from MedImgAI import models


class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Image
        fields = "__all__"

    def validate(self, attrs):
        projectID = attrs.get("projectID")
        dataFileID = attrs.get("dataFileID")

        if models.Image.objects.filter(projectID=projectID, dataFileID=dataFileID):
            raise ValidationError(
                {
                    "msg": "Upload failed! The Data File ID has been existed in this project."
                }
            )

        return attrs


class TeamSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Team
        fields = "__all__"


class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Project
        fields = "__all__"

class DatabaseSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Database
        fields = "__all__"
