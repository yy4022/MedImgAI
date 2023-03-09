import os.path
import subprocess
import uuid

from django.http import FileResponse, HttpRequest, HttpResponse, JsonResponse
from rest_framework import exceptions, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.viewsets import ModelViewSet

from GibbsUNetAPI import settings
from MedImgAI import models, serializers
from MedImgAI.Authentication import TeamAuthentication, UserAuthentication
from MedImgAI.GibbsFramework.IOImage import IOImage
from MedImgAI.GibbsFramework.LoadModel import LoadModel
from MedImgAI.GibbsFramework.Prediction import Prediction
from MedImgAI.GibbsFramework.TrainModel import TrainModel
from MedImgAI.models import Image, Project, Result, Team, TeamInfo, User, Usertoken, Database, Train
from MedImgAI.serializers import (
    ImageSerializer,
    ProjectSerializer,
    TeamSerializer, DatabaseSerializer,
)


def index(request: HttpRequest) -> HttpResponse:
    """
    Greets the requester with `"hello world"`
    Args:
        request: an HTTP request
    Returns:
        an HTTP response
    """
    return HttpResponse("Hello, world!")


class SignUpView(APIView):
    def post(self, request):
        username = request.data["username"]
        password = request.data["password"]
        repeated = request.data["repeated"]
        if repeated != password:
            return Response({"msg": "password is not the same."})
        exist_user = User.objects.filter(username=username, password=password)
        if not exist_user:
            User.objects.create(username=username, password=password)
            return Response({"msg": "Create user success!"})
        else:
            return Response({"msg": "User already exists."})


class LoginView(APIView):
    def post(self, request):
        username = request.data["username"]
        password = request.data["password"]
        exist_user = User.objects.filter(username=username, password=password)
        if not exist_user:
            return Response({"msg": "Login failed! Username or password is incorrect."})
        else:
            token = uuid.uuid4().hex
            Usertoken.objects.update_or_create(username=username, token=token)
            return Response({"msg": "Login success!", "token": token})


class ImageUploadView(ModelViewSet):
    authentication_classes = [TeamAuthentication]

    queryset = models.Image.objects.all()
    serializer_class = serializers.ImageSerializer


class FilterImageView(APIView):
    def post(self, request):
        projectID = request.data["projectID"]
        dataFileID = request.data["dataFileID"]
        exist_image = Result.objects.filter(projectID=projectID, dataFileID=dataFileID)
        if not exist_image:
            return Response(
                {"msg": "Failed! This dataFileID does not exist."},
                status.HTTP_400_BAD_REQUEST,
            )
        else:
            return Response({"projectID": projectID, "dataFileID": dataFileID})


class ImagePreviewView(APIView):
    def get(self, request, projectID, dataFileID):
        exist_dataFile = Result.objects.filter(
            projectID=projectID, dataFileID=dataFileID
        )
        if not exist_dataFile:
            return Response(
                {"msg": "Failed! This dataFileID does not exist."},
                status.HTTP_400_BAD_REQUEST,
            )

        response = FileResponse(open(exist_dataFile.first().previewImage.path, "rb"))
        # response = FileResponse(open(exist_dataFile.first().brainInputImage.path, "rb"))
        return response


class ImageDownloadView(APIView):
    def get(self, request, projectID, dataFileID):
        exist_dataFile = Result.objects.filter(
            projectID=projectID, dataFileID=dataFileID
        )
        if not exist_dataFile:
            return Response(
                {"msg": "Failed! This dataFileID does not exist."},
                status.HTTP_400_BAD_REQUEST,
            )

        response = FileResponse(
            open(exist_dataFile.first().labelOutput3DImage.path, "rb")
        )
        response["content-type"] = "application/octet-stream"
        response["Content-Disposition"] = "attachment; filename=" + os.path.basename(
            exist_dataFile.first().labelOutput3DImage.path
        )
        return response


class ProcessImageView(APIView):
    # Only team members can process the image
    def post(self, request):
        projectID = request.data["projectID"]
        dataFileID = request.data["dataFileID"]

        exist_image = Image.objects.filter(projectID=projectID, dataFileID=dataFileID)
        if not exist_image:
            return Response(
                {"msg": "Failed! This image does not exist."},
                status.HTTP_400_BAD_REQUEST,
            )

        io_arg = IOImage()
        lm_arg = LoadModel()
        pred_arg = Prediction()

        img_obj = Image.objects.filter(projectID=projectID, dataFileID=dataFileID)
        print(img_obj.first().image.path)
        img_path = img_obj.first().image.path

        tensor1, tensor2 = io_arg.readImage(img_path)
        model = lm_arg.loadModel(
            "MedImgAI/MedImgAI/Resources/best_metric_model_180_epoch.pth"
        )
        out_tensor = pred_arg.predictImage(tensor1, model)

        brain_input_path = (
            "MedImgAI/MedImgAI/SaveImage_temp/testOutput/brain_input_image.jpg"
        )
        label_output_path = (
            "MedImgAI/MedImgAI/SaveImage_temp/testOutput/label_output_image.jpg"
        )
        preview_image_path = (
            "MedImgAI/MedImgAI/SaveImage_temp/testOutput/preview_image.jpg"
        )
        label_output_3D_path = (
            "MedImgAI/MedImgAI/SaveImage_temp/testOutput/label_output_3D_image.nii"
        )
        io_arg.saveImage(
            out_tensor,
            tensor2,
            brain_input_path,
            label_output_path,
            preview_image_path,
            label_output_3D_path,
        )

        Result.objects.create(
            projectID=projectID,
            dataFileID=dataFileID,
            brainInputImage="MedImgAI/SaveImage_temp/testOutput/brain_input_image.jpg",
            labelOutputImage="MedImgAI/SaveImage_temp/testOutput/label_output_image.jpg",
            previewImage="MedImgAI/SaveImage_temp/testOutput/preview_image.jpg",
            labelOutput3DImage="MedImgAI/SaveImage_temp/testOutput/label_output_3D_image.nii",
        )
        return Response({"msg": "Process image success!"})


class createTeamView(APIView):
    authentication_classes = [UserAuthentication]

    def post(self, request):
        teamName = request.data["teamName"]
        exist_team = Team.objects.filter(teamName=teamName)
        if not exist_team:
            # "token" --> "teamOwner"
            username = (
                Usertoken.objects.filter(token=request.data["token"]).first().username
            )
            # print(username)
            team_size = request.data["Team Size"]
            is_public = request.data["isPublic"]
            description = request.data["description"]
            Team.objects.create(
                teamName=teamName,
                owner=username,
                scale=team_size,
                isPublic=is_public,
                description=description,
            )
            TeamInfo.objects.create(
                teamName=teamName, userName=username, status="Leader"
            )
            return Response({"msg": "create team success!"})
        else:
            return Response({"msg": "Team has already existed!"}, 409)


class ModifyTeamView(APIView):
    authentication_classes = [UserAuthentication]

    def post(self, request):
        id = request.data["id"]
        originalTeamName = Team.objects.filter(id=id).first().teamName
        TeamName = request.data["teamName"]
        team_size = request.data["Team Size"]
        is_public = request.data["isPublic"]
        description = request.data["description"]

        # Only the leader have authentication to modify team information.
        username = (
            Usertoken.objects.filter(token=request.data.get("token")).first().username
        )
        if_leader = TeamInfo.objects.filter(
            teamName=originalTeamName, userName=username, status="Leader"
        )
        if not if_leader:
            raise exceptions.AuthenticationFailed(
                {"msg": "current user is not leader of the group"}
            )

        # If the teamName has been changed
        if TeamName != originalTeamName:
            # Check if the newTeamName has existed
            exist_teamname = Team.objects.filter(teamName=TeamName)
            if exist_teamname:
                return Response({"msg": "New name has already been used."})

        Team.objects.filter(teamName=originalTeamName).update(
            teamName=TeamName,
            owner=username,
            scale=team_size,
            isPublic=is_public,
            description=description,
        )
        TeamInfo.objects.filter(teamName=originalTeamName).update(
            teamName=TeamName,
        )
        return Response({"msg": "Update success!"})


class TeamListView(APIView):
    authentication_classes = [UserAuthentication]

    def post(self, request):
        username = (
            Usertoken.objects.filter(token=request.data.get("token")).first().username
        )
        teamInfos = Team.objects.none()
        for t in TeamInfo.objects.filter(userName=username).all():
            instance = Team.objects.filter(teamName=t.teamName).first()
            teamInfos |= Team.objects.filter(pk=instance.pk)
            # print(teamInfos)
        teamserializer = TeamSerializer(teamInfos, many=True)
        return Response(teamserializer.data)


class ImageNIIView(APIView):
    def get(self, request):
        exist_dataFile = Image.objects.filter(projectID="project1", dataFileID="data1")
        if not exist_dataFile:
            return Response(
                {"msg": "Failed! This dataFileID does not exist."},
                status.HTTP_400_BAD_REQUEST,
            )

        response = FileResponse(
            open(exist_dataFile.first().image.path, "rb"), as_attachment=False
        )
        return response


class JoinTeamView(APIView):
    authentication_classes = [UserAuthentication]

    def post(self, request):
        team = Team.objects.filter(teamName=request.data.get("teamName")).first()
        if not team:
            return Response({"msg": "Team does not exist!"}, 401)

        isPublic = (
            Team.objects.filter(teamName=request.data.get("teamName")).first().isPublic
        )
        if not isPublic:
            return Response({"msg": "This team is not public!"}, 401)

        userName = (
            Usertoken.objects.filter(token=request.data.get("token")).first().username
        )
        if TeamInfo.objects.filter(userName=userName).first():
            return Response({"msg": "You have already been in this team."}, 401)
        TeamInfo.objects.create(
            teamName=request.data.get("teamName"), userName=userName, status="member"
        )
        return Response({"msg": "Join success!"})


class SearchTeamView(APIView):
    def get(self, request, teamName):
        team = Team.objects.filter(teamName=teamName).first()
        if not team:
            return Response({"msg": "Team does not exist!"}, 401)
        teamserializer = TeamSerializer(team)
        return Response(teamserializer.data)


class FilterTeamView(APIView):
    def post(self, request):
        teamName = request.data["teamName"]
        team = Team.objects.filter(teamName=teamName).first()
        if not team:
            return Response({"msg": "Team does not exist!"}, 401)
        else:
            return Response({"teamName": teamName})


class DeleteMemberView(APIView):
    authentication_classes = [UserAuthentication]

    def post(self, request):
        teamname = request.data["teamName"]
        username = (
            Usertoken.objects.filter(token=request.data.get("token")).first().username
        )
        if_leader = TeamInfo.objects.filter(
            teamName=teamname, userName=username, status="Leader"
        )
        if not if_leader:
            raise exceptions.AuthenticationFailed(
                {"msg": "Current user is not the leader of this team!"}
            )
        memberName = request.data["memberName"]
        if not TeamInfo.objects.filter(teamName=teamname, userName=memberName):
            return Response({"msg": "This member is not in this team."}, 401)
        TeamInfo.objects.filter(teamName=teamname, userName=memberName).delete()
        return Response({"msg": "Delete success!"})


class PreviewSearchView(APIView):
    def get(self, request, projectID, dataFileID):
        head_path = "http://127.0.0.1:8000" + settings.MEDIA_URL
        # Get the path of input image
        input_obj = Image.objects.filter(projectID=projectID, dataFileID=dataFileID)
        input_abs_path = input_obj.first().image.path
        (tempt_input_path, filename_input) = os.path.split(input_abs_path)
        # input_path = os.path.join(head_path, 'MedImgAI', 'SaveImage_temp', filename_input)
        input_path = head_path + "MedImgAI/" + "SaveImage_temp/" + filename_input
        print(input_path)

        # Get the path of output image
        output_obj = Result.objects.filter(projectID=projectID, dataFileID=dataFileID)
        output_abs_path = output_obj.first().labelOutput3DImage.path
        (tempt_output_path, filename_output) = os.path.split(output_abs_path)
        # output_path = os.path.join(head_path, 'MedImgAI', 'SaveImage_temp', 'testOutput', filename_output)
        output_path = (
            head_path
            + "MedImgAI/"
            + "SaveImage_temp/"
            + "testOutput/"
            + filename_output
        )
        print(output_path)

        return JsonResponse({"input_path": input_path, "output_path": output_path})


class ProjectCreateView(APIView):
    authentication_classes = [UserAuthentication]

    def post(self, request):
        projectID = request.data["projectName"]
        teamName = request.data["teamName"]
        isPublic = request.data["isPublic"]
        description = request.data["description"]
        username = (
            Usertoken.objects.filter(token=request.data.get("token")).first().username
        )

        # Check if projectID has been existed
        if Project.objects.filter(projectID=projectID):
            return Response({"msg": "This project name has been existed!"}, 401)

        # Check if this team has been existed
        if not Team.objects.filter(teamName=teamName):
            return Response({"msg": "This team does not exist!"}, 401)

        # Check if current user is the leader of this team
        if_leader = TeamInfo.objects.filter(
            teamName=teamName, userName=username, status="Leader"
        )
        if not if_leader:
            raise exceptions.AuthenticationFailed(
                {"msg": "Current user is not the leader of this team!"}
            )

        # Otherwise, create a new project
        Project.objects.create(
            projectID=projectID,
            teamName=teamName,
            isPublic=isPublic,
            description=description,
        )
        return Response({"msg": "create project success!"})


class DatabaseShowView(APIView):
    # Firstly, check if the user has been login
    authentication_classes = [UserAuthentication]

    def post(self, request):
        projectID = request.data["projectName"]
        if not Project.objects.filter(projectID=projectID).first():
            return Response({"msg": "This project does not exist!"}, 401)

        for i in Image.objects.filter(projectID=projectID).all():
            exist_output = Result.objects.filter(projectID=projectID, dataFileID=i.dataFileID).first()
            if exist_output:
                ifProcessed = "Yes"
            else:
                ifProcessed = "No"

            if not Database.objects.filter(projectID=projectID, dataFileID=i.dataFileID).first():
                Database.objects.create(
                    teamName=Project.objects.filter(projectID=projectID).first().teamName,
                    projectID=projectID,
                    dataFileID=i.dataFileID,
                    ifProcessed=ifProcessed,
                )
            else:
                if Database.objects.filter(projectID=projectID, dataFileID=i.dataFileID).first().ifProcessed != ifProcessed:
                    Database.objects.filter(projectID=projectID, dataFileID=i.dataFileID).delete()
                    Database.objects.create(
                        teamName=Project.objects.filter(projectID=projectID).first().teamName,
                        projectID=projectID,
                        dataFileID=i.dataFileID,
                        ifProcessed=ifProcessed,
                    )


        data_objs = Database.objects.filter(projectID=projectID).all()
        databaseSerializer = DatabaseSerializer(data_objs, many=True)
        return Response(databaseSerializer.data)


class DeleteInputView(APIView):
    authentication_classes = [UserAuthentication]

    def post(self, request):
        projectID = request.data["projectName"]
        dataFileID = request.data["dataFileID"]
        username = (
            Usertoken.objects.filter(token=request.data.get("token")).first().username
        )

        # Only the team leader can delete image.
        teamName = Project.objects.filter(projectID=projectID).first().teamName
        if_leader = TeamInfo.objects.filter(
            teamName=teamName, userName=username, status="Leader"
        )
        if not if_leader:
            return Response(
                {"msg": "Current user has no authentication to delete data!"}, 401
            )

        Image.objects.filter(
            projectID=projectID, dataFileID=dataFileID
        ).first().delete()
        return Response({"msg": "Delete image successfully!"})


class DeleteOutputView(APIView):
    authentication_classes = [UserAuthentication]

    def post(self, request):
        projectID = request.data["projectName"]
        dataFileID = request.data["dataFileID"]
        username = (
            Usertoken.objects.filter(token=request.data.get("token")).first().username
        )

        # Only the team leader can delete image.
        teamName = Project.objects.filter(projectID=projectID).first().teamName
        if_leader = TeamInfo.objects.filter(
            teamName=teamName, userName=username, status="Leader"
        )
        if not if_leader:
            return Response(
                {"msg": "Current user has no authentication to delete data!"}, 401
            )

        Result.objects.filter(
            projectID=projectID, dataFileID=dataFileID
        ).first().delete()
        return Response({"msg": "Delete image successfully!"})


class ProjectListView(APIView):
    authentication_classes = [UserAuthentication]

    def post(self, request):
        username = (
            Usertoken.objects.filter(token=request.data.get("token")).first().username
        )
        exist_team = TeamInfo.objects.filter(userName=username)
        if not exist_team:
            Response({"msg": "Current user belongs to no team!"}, 401)

        projectObjects = Project.objects.none()
        for t in TeamInfo.objects.filter(userName=username).all():
            if Project.objects.filter(teamName=t.teamName).first():
                for p in Project.objects.filter(teamName=t.teamName).all():
                    project_obj = Project.objects.filter(projectID=p.projectID).first()
                    projectObjects |= Project.objects.filter(pk=project_obj.pk)

        projectserializer = ProjectSerializer(projectObjects, many=True)
        return Response(projectserializer.data)


class TrainModelView(APIView):
    authentication_classes = [UserAuthentication]

    def post(self, request):
        projectID = request.data["projectName"]
        alpha = float(request.data["alpha"])
        epoch = int(request.data["epoch"])

        # Check if this project has been existed
        exist_project = Project.objects.filter(projectID=projectID)
        if not exist_project:
            return Response({"msg": "This project does not exist!"}, 401)

        # Check if the input of alpha is right
        print(alpha)
        print(type(alpha))
        if (alpha > 1) or (alpha < 0):
            return Response({"msg": "The input of alpha is illegal! NOTE: alpha range is [0-1]"}, 401)

        # Check if the input of epoch is right
        if epoch <= 0:
            return Response({"msg": "The input of epoch is illegal! NOTE: epoch > 0"}, 401)

        tm_obj = TrainModel()
        port, tb_process, train_log_dir = tm_obj.train_model(alpha, epoch)
        print(tb_process)
        print(type(tb_process))
        # tb_process.kill()
        # new_process = subprocess.Popen()
        new_process = subprocess.Popen(f"tensorboard --host=127.0.0.1 --logdir {train_log_dir} --port={port}", shell=True)
        # print(new_process)
        # print(type(new_process))
        new_process.kill()

        Train.objects.create(
            projectID=projectID,
            alpha=alpha,
            epoch=epoch,
        )

        tensorboard_url = "http://127.0.0.1:" + port

        return Response({"msg": "Train model successfully!",
                         "tensorboardURL": tensorboard_url})
