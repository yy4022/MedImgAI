import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from monai.transforms import Activations, AsDiscrete, Compose, NormalizeIntensity
from PIL import Image

matplotlib.use("agg")

transform = Compose(
    [
        NormalizeIntensity(nonzero=True, channel_wise=True),
    ]
)


class IOImage:
    def readImage(self, filepath):
        """Read image from the given filepath and transform it into the tensors.

        Args:
            filepath: the absolute filepath of the image

        Returns:
            img3d_tensor2, img3d_arr2tensor1: two tensors transformed from the image
        """
        img3d = nib.load(filepath)
        img3d_arr = img3d.get_fdata()
        img3d_arr = np.squeeze(img3d_arr)
        device = torch.device("cpu")
        img3d_tensor = torch.tensor(img3d_arr).to(device)
        img3d_tensor1 = img3d_tensor.permute(3, 0, 1, 2)
        img3d_tensor1 = img3d_tensor1.to(torch.float32)

        # Normalization
        img3d_tensor_normal = transform(img3d_tensor1)
        img3d_tensor2 = img3d_tensor_normal.unsqueeze(0).to(device)
        img3d_arr1 = np.squeeze(img3d_arr)

        img3d_arr2tensor = torch.tensor(img3d_arr1).to(device)
        img3d_arr2tensor1 = img3d_arr2tensor.permute(3, 0, 1, 2)
        return img3d_tensor2, img3d_arr2tensor1

    def saveImage(
        self,
        val_output,
        img3d_arr2tensor,
        brain_input_path,
        label_output_path,
        preview_image_path,
        label_output_3D_path,
    ):
        """

        Args:
            val_output: the label output image
            img3d_arr2tensor: the brain input image of tensor format
            brain_input_path: the filepath for reserving the brain input path
            label_output_path: the filepath for reserving the label output path

        Returns:
            True: indicates that the implementation has been done
        """
        post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        val_output = post_trans(val_output[0])
        plt.figure("image", (24, 6))
        # Save the 2D input brain image
        for i in range(4):
            plt.subplot(1, 4, i + 1)
            plt.title(f"image channel {i}")
            plt.imshow(img3d_arr2tensor[i, :, :, 70].detach().cpu(), cmap="gray")
            plt.savefig(brain_input_path)
        # visualize the 3 channels model output corresponding to this image
        plt.figure("output", (18, 6))

        # Save the 2D output label image
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(f"output channel {i}")
            plt.imshow(val_output[i, :, :, 70].detach().cpu())
            plt.savefig(label_output_path)

        # Save the combination of the above two 2D images
        img1 = Image.open(brain_input_path)
        img2 = Image.open(label_output_path)
        result = Image.new(img1.mode, (1200 * 2, 1200))
        result.paste(img1, box=(0, 0))
        result.paste(img2, box=(200, 600))
        result.save(preview_image_path)

        img_array = np.transpose(val_output.numpy(), (1, 2, 3, 0))
        img_array = img_array / np.max(img_array)
        gray_img = np.zeros((240, 240, 155))
        gray_img[(img_array[..., 0] > 0)] = 3
        gray_img[(img_array[..., 1] > 0)] = 1
        gray_img[(img_array[..., 2] > 0)] = 2
        affine = np.eye(4)
        header = nib.Nifti1Header()
        new_img = nib.Nifti1Image(gray_img, affine, header)
        new_img.to_filename(label_output_3D_path)

        return True
