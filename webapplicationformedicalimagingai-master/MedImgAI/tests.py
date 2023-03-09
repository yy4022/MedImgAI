import os
import shutil
from datetime import datetime

import torch
from django.test import Client, TestCase
from rest_framework.test import APIClient

from MedImgAI.GibbsFramework.IOImage import IOImage
from MedImgAI.GibbsFramework.LoadModel import LoadModel
from MedImgAI.GibbsFramework.Prediction import Prediction
from MedImgAI.GibbsFramework.TrainModel import TrainModel


# class MLTest(TestCase):
#     def test_read_image(self):
#         # Test if it can read image in correct format.
#         current_path = os.path.dirname(__file__)
#         test_filepath = os.path.join(
#             current_path, "MedImgAI/Resources", "BRATS_244.nii.gz"
#         )
#         my_alg = IOImage()
#         img3d_tensor, img3d_arr2tensor = my_alg.readImage(test_filepath)
#         self.assertEqual(img3d_tensor.shape, torch.Size([1, 4, 240, 240, 155]))
#         self.assertEqual(img3d_arr2tensor.shape, torch.Size([4, 240, 240, 155]))
#
#     def test_load_model(self):
#         # Test if it can load the model successfully.
#         current_path = os.path.dirname(__file__)
#         model_filepath = os.path.join(
#             current_path, "MedImgAI/Resources", "best_metric_model_180_epoch.pth"
#         )
#         my_alg = LoadModel()
#         model = my_alg.loadModel(model_filepath)
#         self.assertIsNotNone(model)
#
#     def test_predict_image(self):
#         # Test if it can predict image successfully.
#         # Step1: load the unprocessed image.
#         current_path = os.path.dirname(__file__)
#         test_filepath = os.path.join(
#             current_path, "MedImgAI/Resources", "BRATS_244.nii.gz"
#         )
#         my_alg1 = IOImage()
#         image_tensor, img3d_arr2tensor = my_alg1.readImage(test_filepath)
#
#         # Step2: load the model.
#         model_filepath = os.path.join(
#             current_path, "MedImgAI/Resources", "best_metric_model_180_epoch.pth"
#         )
#         my_alg2 = LoadModel()
#         model = my_alg2.loadModel(model_filepath)
#
#         # Step3: predict the image.
#         my_alg3 = Prediction()
#         val_output = my_alg3.predictImage(image_tensor, model)
#         self.assertIsNotNone(val_output)
#
#     def test_save_image(self):
#         # Test if it can save the output image successfully.
#         # Step1: load the unprocessed image.
#         current_path = os.path.dirname(__file__)
#         test_filepath = os.path.join(
#             current_path, "MedImgAI/Resources", "BRATS_244.nii.gz"
#         )
#         my_alg1 = IOImage()
#         image_tensor, img3d_arr2tensor = my_alg1.readImage(test_filepath)
#
#         # Step2: load the model.
#         model_filepath = os.path.join(
#             current_path, "MedImgAI/Resources", "best_metric_model_180_epoch.pth"
#         )
#         my_alg2 = LoadModel()
#         model = my_alg2.loadModel(model_filepath)
#
#         # Step3: predict the image.
#         my_alg3 = Prediction()
#         val_output = my_alg3.predictImage(image_tensor, model)
#
#         # Step4: save the output image.
#         save_directory = os.path.join(
#             current_path, "MedImgAI/SaveImage_temp", "testOutput"
#         )
#         if os.path.exists(save_directory):
#             shutil.rmtree(save_directory)
#         os.mkdir(save_directory)
#         brain_input_path = os.path.join(save_directory, "brain_input_image.jpg")
#         label_output_path = os.path.join(save_directory, "label_output_image.jpg")
#         preview_image_path = os.path.join(save_directory, "preview_image.jpg")
#         self.assertTrue(
#             my_alg1.saveImage(
#                 val_output,
#                 img3d_arr2tensor,
#                 brain_input_path,
#                 label_output_path,
#                 preview_image_path,
#             )
#         )
#         self.assertTrue(os.path.exists(brain_input_path))
#         self.assertTrue(os.path.exists(label_output_path))
#         self.assertTrue(os.path.exists(preview_image_path))
#
#     def test_medimgai_url(self):
#         c = Client()
#         # response = c.post('/med-img-ai/')
#         # response.status_code
#         response = c.get("/med-img-ai/")
#         # print(response.status_code)
#         # print(response.content)
#         self.assertEqual(response.status_code, 200)
#         self.assertEqual(response.content, b"Hello, world!")
#
#
# class SignUpURLTest(TestCase):
#     def test_signup_successfully(self):
#         c = APIClient()
#         data = {"username": "yy01", "password": 12345, "repeated": 12345}
#         response = c.post("/signup/", data)
#         self.assertEqual(response.status_code, 200)
#         self.assertEqual(response.json()["msg"], "Create user success!")
#
#     def test_existed_username(self):
#         c = APIClient()
#         data = {"username": "yy01", "password": 12345, "repeated": 12345}
#         c.post("/signup/", data)
#         response = c.post("/signup/", data)
#         self.assertEqual(response.status_code, 200)
#         self.assertEqual(response.json()["msg"], "User already exists.")
#
#     def test_password_error(self):
#         c = APIClient()
#         data = {"username": "yy01", "password": 12345, "repeated": 54321}
#         response = c.post("/signup/", data)
#         self.assertEqual(response.status_code, 200)
#         self.assertEqual(response.json()["msg"], "password is not the same.")
#
#
# class LoginURLTest(TestCase):
#     def test_login_successfully(self):
#         c = APIClient()
#         data = {"username": "yy01", "password": 12345, "repeated": 12345}
#         c.post("/signup/", data)
#         response = c.post("/login/", data)
#         self.assertEqual(response.status_code, 200)
#         self.assertEqual(response.json()["msg"], "Login success!")
#
#     def test_login_error(self):
#         c = APIClient()
#         data = {"username": "yy01", "password": 12345, "repeated": 12345}
#         response = c.post("/login/", data)
#         self.assertEqual(response.status_code, 200)
#         self.assertEqual(
#             response.json()["msg"], "Login failed! Username or password is incorrect."
#         )
#
#
# class UploadURLTest(TestCase):
#     def test_upload_successfully(self):
#         c = APIClient()
#         current_path = os.path.dirname(__file__)
#         test_filepath = os.path.join(
#             current_path, "MedImgAI/Resources", "BRATS_244.nii.gz"
#         )
#         time = datetime.now()
#         with open(test_filepath, "rb") as image:
#             data = {
#                 "projectID": "Project_test",
#                 "dataFileID": "Data_test",
#                 "image": image,
#                 "addTime": time,
#             }
#             response = c.post("/upload/", data)
#             self.assertEqual(response.status_code, 201)
#
#     def test_repeated_name(self):
#         c = APIClient()
#         current_path = os.path.dirname(__file__)
#         test_filepath = os.path.join(
#             current_path, "MedImgAI/Resources", "BRATS_244.nii.gz"
#         )
#         time = datetime.now()
#         with open(test_filepath, "rb") as image:
#             data = {
#                 "projectID": "Project_test",
#                 "dataFileID": "Data_test",
#                 "image": image,
#                 "addTime": time,
#             }
#             c.post("/upload/", data)
#
#         with open(test_filepath, "rb") as image:
#             data = {
#                 "projectID": "Project_test",
#                 "dataFileID": "Data_test",
#                 "image": image,
#                 "addTime": time,
#             }
#             response = c.post("/upload/", data)
#             self.assertEqual(response.status_code, 400)
#             self.assertEqual(
#                 response.json()["msg"],
#                 ["Upload failed! The Data File ID has been existed in this project."],
#             )
#
#     def test_empty_projectID(self):
#         c = APIClient()
#         current_path = os.path.dirname(__file__)
#         test_filepath = os.path.join(
#             current_path, "MedImgAI/Resources", "BRATS_244.nii.gz"
#         )
#         time = datetime.now()
#         with open(test_filepath, "rb") as image:
#             data = {"dataFileID": "Data_test", "image": image, "addTime": time}
#             response = c.post("/upload/", data)
#             self.assertEqual(response.status_code, 400)
#             self.assertEqual(response.json()["projectID"], ["This field is required."])
#
#     def test_empty_dataFileID(self):
#         c = APIClient()
#         current_path = os.path.dirname(__file__)
#         test_filepath = os.path.join(
#             current_path, "MedImgAI/Resources", "BRATS_244.nii.gz"
#         )
#         time = datetime.now()
#         with open(test_filepath, "rb") as image:
#             data = {"projectID": "Project_test", "image": image, "addTime": time}
#             response = c.post("/upload/", data)
#             self.assertEqual(response.status_code, 400)
#             self.assertEqual(response.json()["dataFileID"], ["This field is required."])
#
#     def test_empty_image(self):
#         c = APIClient()
#         time = datetime.now()
#         data = {"projectID": "Project_test", "dataFileID": "Data_test", "addTime": time}
#         response = c.post("/upload/", data)
#         self.assertEqual(response.status_code, 400)
#         self.assertEqual(response.json()["image"], ["No file was submitted."])
#
#
# class ImplementURLtest(TestCase):
#     def test_inexisted_image(self):
#         c = APIClient()
#         data1 = {"projectID": "Project_test", "dataFileID": "Data_test"}
#         response = c.post("/implement/", data1)
#         self.assertEqual(response.status_code, 400)
#         self.assertEqual(response.json()["msg"], "Failed! This image does not exist.")
#
#     def test_process_sucessful(self):
#         c = APIClient()
#         current_path = os.path.dirname(__file__)
#         test_filepath = os.path.join(
#             current_path, "MedImgAI/Resources", "BRATS_244.nii.gz"
#         )
#         time = datetime.now()
#         with open(test_filepath, "rb") as image:
#             data = {
#                 "projectID": "Project_test",
#                 "dataFileID": "Data_test",
#                 "image": image,
#                 "addTime": time,
#             }
#             image_filepath = os.path.join(
#                 current_path, "MedImgAI/SaveImage_temp", "BRATS_244.nii.gz"
#             )
#             os.remove(image_filepath)
#             c.post("/upload/", data)
#
#         data1 = {"projectID": "Project_test", "dataFileID": "Data_test"}
#         response = c.post("/implement/", data1)
#         self.assertEqual(response.status_code, 200)
#         self.assertEqual(response.json()["msg"], "Process image success!")
#
#
# class FilterURLtest(TestCase):
#     def test_filter_sucessful(self):
#         c = APIClient()
#         current_path = os.path.dirname(__file__)
#         test_filepath = os.path.join(
#             current_path, "MedImgAI/Resources", "BRATS_244.nii.gz"
#         )
#         time = datetime.now()
#         with open(test_filepath, "rb") as image:
#             data = {
#                 "projectID": "Project_test",
#                 "dataFileID": "Data_test",
#                 "image": image,
#                 "addTime": time,
#             }
#             image_filepath = os.path.join(
#                 current_path, "MedImgAI/SaveImage_temp", "BRATS_244.nii.gz"
#             )
#             os.remove(image_filepath)
#             c.post("/upload/", data)
#
#         data1 = {"projectID": "Project_test", "dataFileID": "Data_test"}
#         c.post("/implement/", data1)
#         response = c.post("/filter/", data1)
#         self.assertEqual(response.status_code, 200)
#         self.assertEqual(
#             response.json(), {"projectID": "Project_test", "dataFileID": "Data_test"}
#         )
#
#     def test_inexisted_image(self):
#         c = APIClient()
#         data1 = {"projectID": "Project_test", "dataFileID": "Data_test"}
#         response = c.post("/filter/", data1)
#         self.assertEqual(response.status_code, 400)
#         self.assertEqual(
#             response.json()["msg"], "Failed! This dataFileID does not exist."
#         )
#
#
class PreviewURLtest(TestCase):
    def test_preview_sucessful(self):
        c = APIClient()
        current_path = os.path.dirname(__file__)
        # test_filepath = os.path.join(
        #     current_path, "MedImgAI/Resources", "BRATS_244.nii.gz"
        # )
        test_filepath = "D:\Code\webapplicationformedicalimagingai\MedImgAI\MedImgAI\Resources\BRATS_244.nii.gz"
        time = datetime.now()
        with open(test_filepath, "rb") as image:
            data = {
                "projectID": "Project_test",
                "dataFileID": "Data_test",
                "image": image,
                "addTime": time,
            }
            # image_filepath = os.path.join(
            #     current_path, "MedImgAI/MedImgAI/SaveImage_temp", "BRATS_244.nii.gz"
            # )
            # os.remove(image_filepath)
            c.post("/upload/", data)

        data1 = {"projectID": "Project_test", "dataFileID": "Data_test"}
        c.post("/implement/", data1)
        response = c.get("/preview/Project_test/Data_test")
        self.assertEqual(response.status_code, 200)
#
#     def test_inexisted_image(self):
#         c = APIClient()
#         response = c.get("/preview/Project_test/Data_test")
#         self.assertEqual(response.status_code, 400)
#         self.assertEqual(
#             response.json()["msg"], "Failed! This dataFileID does not exist."
#         )
#
#
# class DownloadURLtest(TestCase):
#     def test_download_sucessful(self):
#         c = APIClient()
#         current_path = os.path.dirname(__file__)
#         test_filepath = os.path.join(
#             current_path, "MedImgAI/Resources", "BRATS_244.nii.gz"
#         )
#         time = datetime.now()
#         with open(test_filepath, "rb") as image:
#             data = {
#                 "projectID": "Project_test",
#                 "dataFileID": "Data_test",
#                 "image": image,
#                 "addTime": time,
#             }
#             image_filepath = os.path.join(
#                 current_path, "MedImgAI/SaveImage_temp", "BRATS_244.nii.gz"
#             )
#             os.remove(image_filepath)
#             c.post("/upload/", data)
#
#         data1 = {"projectID": "Project_test", "dataFileID": "Data_test"}
#         c.post("/implement/", data1)
#         response_input = c.get("/download/Project_test/Data_test/input")
#         response_output = c.get("/download/Project_test/Data_test/output")
#         self.assertEqual(response_input.status_code, 200)
#         self.assertEqual(response_output.status_code, 200)
#
#     def test_inexisted_image(self):
#         c = APIClient()
#         response_input = c.get("/download/Project_test/Data_test/input")
#         response_output = c.get("/download/Project_test/Data_test/output")
#         self.assertEqual(response_input.status_code, 400)
#         self.assertEqual(response_output.status_code, 400)
#         self.assertEqual(
#             response_input.json()["msg"], "Failed! This dataFileID does not exist."
#         )
#         self.assertEqual(
#             response_output.json()["msg"], "Failed! This dataFileID does not exist."
#         )
#
#     def test_illegal_imageName(self):
#         c = APIClient()
#         current_path = os.path.dirname(__file__)
#         test_filepath = os.path.join(
#             current_path, "MedImgAI/Resources", "BRATS_244.nii.gz"
#         )
#         time = datetime.now()
#         with open(test_filepath, "rb") as image:
#             data = {
#                 "projectID": "Project_test",
#                 "dataFileID": "Data_test",
#                 "image": image,
#                 "addTime": time,
#             }
#             image_filepath = os.path.join(
#                 current_path, "MedImgAI/SaveImage_temp", "BRATS_244.nii.gz"
#             )
#             os.remove(image_filepath)
#             c.post("/upload/", data)
#
#         data1 = {"projectID": "Project_test", "dataFileID": "Data_test"}
#         c.post("/implement/", data1)
#         response = c.get("/download/Project_test/Data_test/IllegalName")
#         self.assertEqual(response.status_code, 400)
#         self.assertEqual(response.json()["msg"], "Failed! This imageName is illegal.")

# class TrainTest(TestCase):
#     def test_train_model(self):
#         my_alg = TrainModel()
#         my_alg.train_model(0.3, 3)


    # def test_input(self):
    #     alpha = "1/3"
