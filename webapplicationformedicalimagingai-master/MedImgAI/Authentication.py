from rest_framework import exceptions
from rest_framework.authentication import BaseAuthentication

from MedImgAI.models import Project, TeamInfo, Usertoken


class UserAuthentication(BaseAuthentication):
    def authenticate(self, request):
        token = request.data.get("token")
        if not token:
            raise exceptions.AuthenticationFailed({"msg": "Please login firstly."})
        else:
            exist_user = Usertoken.objects.filter(token=token)
            if not exist_user:
                raise exceptions.AuthenticationFailed(
                    {"msg": "User information incorrect."}
                )


class TeamAuthentication(BaseAuthentication):
    def authenticate(self, request):
        token = request.data.get("token")
        projectID = request.data.get("projectID")
        if not token:
            raise exceptions.AuthenticationFailed({"msg": "Please login firstly."})
        else:
            exist_user = Usertoken.objects.filter(token=token)
            if not exist_user:
                raise exceptions.AuthenticationFailed(
                    {"msg": "User information incorrect."}
                )
            else:
                exist_projectID = Project.objects.filter(projectID=projectID)
                if not exist_projectID:
                    raise exceptions.AuthenticationFailed(
                        {"msg": "This project does not exist."}
                    )
                teamName = Project.objects.filter(projectID=projectID).first().teamName
                username = (
                    Usertoken.objects.filter(token=request.data.get("token"))
                    .first()
                    .username
                )
                if not TeamInfo.objects.filter(
                    teamName=teamName, userName=username
                ).first():
                    raise exceptions.AuthenticationFailed(
                        {
                            "msg": "Current user has no authentication to upload. (Only guest for this project)"
                        }
                    )
