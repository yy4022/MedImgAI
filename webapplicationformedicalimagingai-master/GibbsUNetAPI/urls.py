"""GibbsUNetAPI URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.contrib import admin
from django.urls import include, path, re_path
from django.views.static import serve
from rest_framework.routers import DefaultRouter

from MedImgAI.views import (
    DeleteInputView,
    DeleteMemberView,
    DeleteOutputView,
    FilterImageView,
    FilterTeamView,
    ImageDownloadView,
    ImagePreviewView,
    ImageUploadView,
    JoinTeamView,
    LoginView,
    ModifyTeamView,
    PreviewSearchView,
    ProcessImageView,
    ProjectCreateView,
    ProjectListView,
    SearchTeamView,
    SignUpView,
    TeamListView,
    createTeamView,
    DatabaseShowView,
    TrainModelView,
)

router = DefaultRouter()
router.register(r"upload", ImageUploadView)

urlpatterns = [
    re_path(r"MedImgAI/(?P<path>.*)$", serve, {"document_root": settings.MEDIA_ROOT}),
    path("admin/", admin.site.urls),
    path("med-img-ai/", include("MedImgAI.urls")),
    path("signup/", SignUpView.as_view()),
    path("login/", LoginView.as_view()),
    path("", include(router.urls)),
    path("implement/", ProcessImageView.as_view()),
    path("filter/", FilterImageView.as_view()),
    path("team/create/", createTeamView.as_view()),
    re_path(
        "preview/(?P<projectID>.*)/(?P<dataFileID>.*)$", ImagePreviewView.as_view()
    ),
    re_path(
        "download/(?P<projectID>.*)/(?P<dataFileID>.*)$",
        ImageDownloadView.as_view(),
    ),
    # re_path("image/(?P<projectID>.*)/(?P<dataFileID>.*)$", ImageNIIView.as_view()),
    # path("image/", ImageNIIView.as_view()),
    path("team/show/", TeamListView.as_view()),
    path("team/join/", JoinTeamView.as_view()),
    # re_path("team/modify/(?P<teamname>.*)$", ModifyTeamView.as_view()),
    path("team/modify/", ModifyTeamView.as_view()),
    re_path("team/search/(?P<teamName>.*)$", SearchTeamView.as_view()),
    path("filter/team/", FilterTeamView.as_view()),
    path("team/delete/", DeleteMemberView.as_view()),
    re_path(
        "preview3D/(?P<projectID>.*)/(?P<dataFileID>.*)$", PreviewSearchView.as_view()
    ),
    path("project/create/", ProjectCreateView.as_view()),
    path("database/show/", DatabaseShowView.as_view()),
    path("inputData/delete/", DeleteInputView.as_view()),
    path("outputData/delete/", DeleteOutputView.as_view()),
    path("project/show/", ProjectListView.as_view()),
    path("train/model/", TrainModelView.as_view()),
]
