# gradcam.py
import torch
import numpy as np
from PIL import Image
import matplotlib.cm as cm

class GradCAM:
    """
    Simple Grad-CAM for CNNs (works with ResNet-style models).
    Usage:
        cam = GradCAM(model, target_layer)
        mask = cam.generate(input_tensor, target_class)  # mask shape (H, W), values 0..1
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output  # tensor
            # ensure output requires grad for hook registration
            output.requires_grad_(True)
            # register a hook on the activations to capture gradients in backward
            output.register_hook(self._save_gradients)
        self.target_layer.register_forward_hook(forward_hook)

    def _save_gradients(self, grad):
        self.gradients = grad

    def generate(self, input_tensor, target_class=None):
        """
        input_tensor: Tensor shape (1, C, H, W) (requires grad=False)
        returns: cam_mask numpy array shape (H, W) normalized 0..1
        """
        # ensure gradients are zeroed
        self.model.zero_grad()
        out = self.model(input_tensor)  # forward
        if target_class is None:
            target_class = out.argmax(dim=1).item()
        # scalar score for the target class
        score = out[:, target_class]
        # backward to compute gradients wrt activations
        score.backward(retain_graph=True)
        # get activations & gradients
        activations = self.activations.detach()        # (1, C, h, w)
        gradients = self.gradients.detach()            # (1, C, h, w)
        # global-average-pool gradients -> weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        # weighted combination of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1,1,h,w)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()  # (h, w)
        # normalize to 0..1
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / (cam.max() + 1e-8)
        return cam

def overlay_cam_on_image(pil_img, cam_mask, alpha=0.5, colormap='jet'):
    """
    pil_img: PIL.Image (grayscale or RGB)
    cam_mask: numpy array HxW with values 0..1
    alpha: blend factor for heatmap
    returns PIL.Image RGB of overlay
    """
    # resize cam_mask to image size
    img_w, img_h = pil_img.size
    cam_img = Image.fromarray(np.uint8(cam_mask * 255)).resize((img_w, img_h), resample=Image.BILINEAR)
    cam = np.array(cam_img) / 255.0  # HxW 0..1
    # create heatmap (H x W x 3)
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(cam)[:, :, :3]  # ignore alpha
    heatmap = np.uint8(heatmap * 255)

    # convert pil image to rgb array
    base = pil_img.convert('RGB')
    base_arr = np.array(base).astype(np.uint8)

    # blend
    overlay_arr = (heatmap * alpha + base_arr * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(overlay_arr)
