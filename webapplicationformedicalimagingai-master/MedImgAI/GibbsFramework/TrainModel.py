import datetime
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import glob
from monai.data import CacheDataset, DataLoader
from typing import Optional
import torch.nn as nn
from monai.networks.nets import UNet #monai
from monai.losses import DiceLoss

import subprocess
import random

from monai.transforms import (
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                torch.logical_or(
                    torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d


class Fourier:
    """
    Helper class storing Fourier mappings
    """

    @staticmethod
    def shift_fourier(x: torch.Tensor, n_dims: int) -> torch.tensor:
        """
        Applies fourier transform and shifts the zero-frequency component to the
        center of the spectrum. Only the spatial dimensions get transformed.

        Args:
            x: image to transform.
            n_dims: number of spatial dimensions.
        Returns
            k: k-space data.
        """
        k: torch.tensor = torch.fft.fftshift(
            torch.fft.fftn(x, dim=tuple(range(-n_dims, 0))),
            dim=tuple(range(-n_dims, 0)))
        # print(tuple(range(-n_dims, 0)))
        return k

    @staticmethod
    def inv_shift_fourier(k: torch.Tensor, n_dims: int) -> torch.tensor:
        """
        Applies inverse shift and fourier transform. Only the spatial
        dimensions are transformed.

        Args:
            k: k-space data.
            n_dims: number of spatial dimensions.
        Returns:
            x: tensor in image space.
        """
        x: torch.tensor = torch.fft.ifftn(
            torch.fft.ifftshift(k, dim=tuple(range(-n_dims, 0))),
            dim=tuple(range(-n_dims, 0))).real
        return x


class GibbsNoiseLayer(nn.Module, Fourier):
    """
    The layer applies Gibbs noise to 2D/3D images.

    Args:
        alpha: Parametrizes the intensity of the Gibbs noise filter applied.
            Takes values in the interval [0,1] with alpha = 1.0 acting as the
            identity mapping.
    """

    def __init__(self, alpha: Optional[float] = None) -> None:

        nn.Module.__init__(self)

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')  # GPU optional

        if alpha is None:
            self.alpha = torch.rand(1, requires_grad=True, device=self.device)
        else:
            alpha = min(max(alpha, 0.), 1.)  # alpha 0-1
            self.alpha = torch.tensor([alpha], requires_grad=True,
                                      device=self.device)

    def forward(self, img: torch.Tensor) -> torch.Tensor:

        n_dims = len(img.shape[1:])
        # FT
        k = self.shift_fourier(img, n_dims)
        # build and apply mask
        k = self._apply_mask(k)
        # map back
        img = self.inv_shift_fourier(k, n_dims)

        return img

    def _apply_mask(self, k: torch.Tensor) -> torch.Tensor:
        """Builds and applies a mask on the spatial dimensions.

        Args:
            k: k-space version of the image.
        Returns:
            masked version of the k-space image.
        """
        shape = k.shape[1:]

        center = (torch.tensor(shape, dtype=torch.float, device=self.device, requires_grad=True) - 1) / 2
        coords = torch.meshgrid(list(torch.linspace(0, i - 1, i) for i in shape))
        # need to subtract center coord and then square for Euc distance
        dist_from_center = torch.sqrt(sum([(coord.to(self.device) - c) ** 2 for (coord, c) in zip(coords, center)]))

        alpha_norm = self.alpha * dist_from_center.max()
        norm_dist = dist_from_center / alpha_norm
        mask = norm_dist.where(norm_dist < 1, torch.zeros_like(alpha_norm))
        mask = mask.where(norm_dist > 1, torch.ones_like(alpha_norm))
        # add channel dimension into mask
        mask = torch.repeat_interleave(mask[None], k.size(0), 0)
        # apply binary mask
        k_masked = k * mask

        return k_masked


class Gibbs_UNet(nn.Module):
    """ResUnet with Gibbs layer"""

    def __init__(self, alpha=Optional[None]):
        super().__init__()

        self.gibbs = GibbsNoiseLayer(alpha)

        self.res_unet = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2
        )

    def forward(self, img):
        img = self.gibbs(img)
        print(img.shape)
        img = self.res_unet(img)
        return img

class TrainModel:
    # def data_process(self):
    #     return

    # define inference method

    def train_model(self, alpha, epoch):
        current_time = str(datetime.datetime.now().timestamp())
        train_log_dir = 'logs/tensorboard/train/' + current_time
        train_summary_writer = SummaryWriter(train_log_dir)

        # data root directory
        data_root = 'MedImgAI/MedImgAI/Resources/Task01_BrainTumor'
        train_images = sorted(glob.glob(os.path.join(data_root, 'imagesTr', '*.nii.gz')))
        train_labels = sorted(glob.glob(os.path.join(data_root, 'labelsTr', '*.nii.gz')))

        # create a training data list
        data_dict = [{'image': str(image_name), 'label': str(label_name)} for image_name, label_name in
                     zip(train_images, train_labels)]

        print(data_dict)

        # define transforms for image and segmentation
        train_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                RandSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64], random_size=False),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )

        # create a training data loader
        train_ds = CacheDataset(data=data_dict, transform=train_transform, cache_rate=1.0)
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = Gibbs_UNet(alpha).to(device) # alpha [0 -1] fraction

        loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
        optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)

        train_summary_writer = SummaryWriter(train_log_dir)


        def epoch_train(num_epoch):
            epoch_loss_values = []

            for epoch in range(num_epoch):
                model.train()
                epoch_loss = 0
                step = 0
                print("-" * 10)
                print(f"epoch {epoch + 1}/{num_epoch}")

                for batch_data in train_loader:
                    step += 1
                    inputs, labels = (
                        batch_data["image"].to(device),
                        batch_data["label"].to(device),
                    )
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                lr_scheduler.step()
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
                train_summary_writer.add_scalar("loss", epoch_loss, epoch)

                train_summary_writer.flush()
            train_summary_writer.close()

        epoch_train(epoch) # epoch
        port = str(random.randint(1000, 7000))
        print(port)
        tb_process = subprocess.Popen(f"tensorboard --host=127.0.0.1 --logdir {train_log_dir} --port={port}", shell=True)
        # print(tb_process)
        # command = f"tensorboard --host=127.0.0.1 --logdir {train_log_dir} --port={port}"
        # subprocess.run(command, shell=True)
        # tb_process.terminate()


        print(train_log_dir)

        return port, tb_process, train_log_dir









