import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class ZeroShotObjectDetector:

    EXAMPLE_MODELS = [
        "IDEA-Research/grounding-dino-tiny",
        "IDEA-Research/grounding-dino-base",
        "openmmlab-community/mm_grounding_dino_large_all",
        "openmmlab-community/mm_grounding_dino_base_all",
        "google/owlv2-base-patch16-ensemble",
        "iSEE-Laboratory/llmdet_tiny",
        "iSEE-Laboratory/llmdet_base",
        "iSEE-Laboratory/llmdet_large",
    ]

    def __init__(self, model_id: str, device):
        """
        model_id: str, e.g. 'IDEA-Research/grounding-dino-tiny'
        device: torch device
        """
        self.device = device
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, local_files_only=True).to(self.device)

    def predict(
            self, 
            image: Image.Image | np.ndarray, 
            text_labels: list[list[str]], 
            threshold=0.3):
        """
        image: input image, array or PIL image
        text_labels: text query of [["my object description"]] format
        threshold: bounding box detection threshold
        """

        if isinstance(image, np.ndarray):
            height, width = image.shape[:2]
        elif isinstance(image, Image.Image):
            height, width = image.height, image.width
         
        inputs = self.processor(images=image, text=text_labels, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=[(height, width)]
        )

        # boxes = results[0]["scores"]
        # boxes = results[0]["boxes"]
        # text_labels = results[0]["text_labels"]
        return results
    

class SegmentorSAM2:

    def __init__(self, model_id):
        """ 
        Try model_id from:
            "facebook/sam2-hiera-tiny"
            "facebook/sam2-hiera-small"
            "facebook/sam2-hiera-base-plus"
            "facebook/sam2-hiera-large"
            "facebook/sam2.1-hiera-tiny"
            "facebook/sam2.1-hiera-small"
            "facebook/sam2.1-hiera-base-plus"
            "facebook/sam2.1-hiera-large"
        """
        from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES
        assert model_id in HF_MODEL_ID_TO_FILENAMES, f"Choose from {list(HF_MODEL_ID_TO_FILENAMES.keys())}"

        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.predictor = SAM2ImagePredictor.from_pretrained(model_id)

    def predict(self, image: np.ndarray | Image.Image, box: np.ndarray):
        """
        image (np.ndarray or PIL Image): The input image to embed in RGB format.
            If numpy array, shape (w,h,3)
            The image should be in HWC format if np.ndarray, or WHC format if PIL Image
            with pixel values in [0, 255].
        box (np.ndarray or None): (nb_boxes,4) XYXY bouding boxes to be used as prompts for sam2

        Return: 
        SAM2 returns 3 mask candidates for all box prompts along with predicted qualities
        masks: torch.Tensor, (nb_boxes, 3, w, h), 3 mask for each box as binary imgs of the same resolution as input image
        qualities: torch.Tensor, (nb_boxes, 3), quality of the masks in [0,1] (1 is best)
        """
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(image)
            masks, qualities, _ = self.predictor.predict(box=box)

        if len(masks.shape) == 3:
            # sam2 outputs shape depends on nb of input boxes
            # add a mask id dimension
            # 1 mask  -> (3, 480, 640), (3,)
            # 2 masks -> (2, 3, 480, 640) (2, 3)
            # -> add mask dimension when only one box detected
            masks = np.expand_dims(masks, 0)
            qualities = np.expand_dims(qualities, 0)

        return masks, qualities


def pad_boxes(boxes: np.ndarray | torch.Tensor, w, h, pixpad=20):
    """
    Add some padding to 

    boxes: np
    """
    # boxes assumed to be in xyxy format
    boxes[:,0] = torch.clamp(boxes[:,0] - pixpad, min=0)
    boxes[:,1] = torch.clamp(boxes[:,1] - pixpad, min=0)
    boxes[:,2] = torch.clamp(boxes[:,2] + pixpad, max=w)
    boxes[:,3] = torch.clamp(boxes[:,3] + pixpad, max=h)
    return boxes