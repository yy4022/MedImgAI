import torch
from monai.inferers import sliding_window_inference


class Prediction:
    def inference(self, input, model):
        VAL_AMP = True

        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=(128, 128, 64),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )

        if VAL_AMP:
            with torch.cuda.amp.autocast():
                return _compute(input)
        else:
            return _compute(input)

    def predictImage(self, img3d_tensor, model):
        """Predict the image with label.

        Args:
            img3d_tensor: the input brain image
            model: the Gibbs model for processing the input image

        Returns:
            val_ouput: the output image with label
        """
        with torch.no_grad():
            # select one image to evaluate and visualize the model output
            val_input = img3d_tensor
            val_output = self.inference(val_input, model)
            return val_output
