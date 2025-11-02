"""
vision_agent.py

A self-contained class for diabetic retinopathy stage detection.
Handles model loading, preprocessing, prediction, and Grad-CAM visualization.
"""

import logging
from typing import Dict, Any, Tuple

import numpy as np
import torch
import cv2
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# Configure logger for module
logger = logging.getLogger("VisionAgent")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[VisionAgent] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class VisionAgent:
    """
    VisionAgent for DR staging using a Hugging Face model.

    Parameters
    ----------
    model_id : str, optional
        Hugging Face model identifier (default: "AsmaaElnagger/Diabetic_RetinoPathy_detection").
    """

    def __init__(self, model_id: str = "AsmaaElnagger/Diabetic_RetinoPathy_detection"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model and processor from {model_id}")
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForImageClassification.from_pretrained(model_id).to(
            self.device
        )
        self.model.eval()
        self.last_conv_layer = self._find_last_conv_layer_name()

        # Internal buffers for Grad-CAM
        self._conv_output = None
        self._gradients = None
        self._register_hooks()

    def _find_last_conv_layer_name(self) -> str:
        """Return the name of the last convolutional layer in the model."""
        modules = list(self.model.named_modules())  # convert generator to list
        for name, module in reversed(modules):
            if isinstance(module, torch.nn.Conv2d):
                return name
        raise ValueError("No Conv2D layer found in model.")

    def _register_hooks(self):
        """Attach forward and backward hooks to capture Grad-CAM data."""

        def save_conv_output(module, inp, out):
            self._conv_output = out

        def save_gradients(module, grad_in, grad_out):
            self._gradients = grad_out[0]

        layer = dict(self.model.named_modules())[self.last_conv_layer]
        layer.register_forward_hook(save_conv_output)
        layer.register_full_backward_hook(save_gradients)

    def _generate_heatmap(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Grad-CAM heatmap with proper upsampling to input resolution.

        Returns
        -------
        heatmap_colored : numpy.ndarray
            Heatmap resized to input image resolution in RGB uint8 form.
        predictions : numpy.ndarray
            Raw model output probabilities.
        """
        self._conv_output, self._gradients = None, None

        # Get input image size from the processed inputs
        # Assuming standard image processor output shape: (batch, channels, height, width)
        input_height, input_width = (
            inputs["pixel_values"].shape[2:] if "pixel_values" in inputs else (224, 224)
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_index = torch.argmax(probs[0]).item()
        score = outputs.logits[:, predicted_index]

        self.model.zero_grad()
        score.backward()

        if self._conv_output is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture data correctly.")

        # Compute channel-wise gradient average
        grads = self._gradients.mean(dim=[2, 3], keepdim=True)

        # Weight activations by gradients
        weighted_activations = (self._conv_output * grads).sum(dim=1, keepdim=True)

        # Apply ReLU to keep only positive contributions
        heatmap = torch.relu(weighted_activations.squeeze()).detach().cpu().numpy()

        # Normalize to [0, 1]
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)

        # Resize heatmap to original input resolution
        heatmap_resized = cv2.resize(
            heatmap, (input_width, input_height), interpolation=cv2.INTER_LINEAR
        )

        # Apply Gaussian blur for smoother transitions and better localization
        heatmap_resized = cv2.GaussianBlur(heatmap_resized, (9, 9), 2)

        # Re-normalize after blur
        heatmap_resized = (heatmap_resized - np.min(heatmap_resized)) / (
            np.max(heatmap_resized) - np.min(heatmap_resized) + 1e-8
        )

        # Apply colormap
        heatmap_colormap = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colormap, cv2.COLOR_BGR2RGB)

        return heatmap_colored, probs.detach().cpu().numpy()

    def predict(self, image_source) -> Dict[str, Any]:
        """
        Predict diabetic retinopathy stage and return Grad-CAM heatmap.

        Parameters
        ----------
        image_source : str
            Path to image or image array.

        Returns
        -------
        dict
            {
                'stage': int,
                'confidence': float,
                'key_regions': numpy.ndarray (RGB heatmap)
            }
        """
        if isinstance(image_source, str):
            image = Image.open(image_source).convert("RGB")
        elif isinstance(image_source, np.ndarray):
            image = Image.fromarray(image_source.astype(np.uint8)).convert("RGB")
        else:
            image = image_source.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")

        try:
            heatmap, predictions = self._generate_heatmap(inputs)
        except Exception as exc:
            logger.exception("Prediction failed: %s", exc)
            raise RuntimeError(f"Prediction failed: {exc}")

        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        image_id = str(uuid.uuid4())

        logger.info(
            f"Processed image_id={image_id} | Stage={predicted_index} | Confidence={confidence:.4f}"
        )

        return {
            "image_id": image_id,
            "stage": predicted_index,
            "confidence": confidence,
            "key_regions": heatmap,
        }
