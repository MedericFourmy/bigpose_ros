import numpy as np
import torch
from happypose.toolbox.renderer.types import Panda3dLightData, BatchRenderOutput



DEFAULT_RENDER_PARAMS = {
    "render_normals": True,
    "render_depth": True,
    "render_binary_mask": True,
}

def get_panda3d_ambient():
    return Panda3dLightData(light_type="ambient", color=(1.0, 1.0, 1.0, 1.0)) 


def render_ts(arr: np.ndarray):
    ts = torch.from_numpy(arr)
    # add batch dim if not present
    if len(ts.shape) == 2:
        ts = ts.unsqueeze(0)
    return ts


def extract_np_from_renderings(renderings: BatchRenderOutput, index: int, fields: list[str] = ["rgb", "depth", "mask", "normals"]):
    """
    renderings, BatchRenderOutput: output of Panda3dBatchRenderer.render, torch tensors with: 
    - BatchRenderOutput.rgbs: rendered color images, normalized (RGB/255) 
    - BatchRenderOutput.depths: rendered depths, same metric as mesh
    - BatchRenderOutput.normals: rendered normals, unit norm 3D vectors 
    - BatchRenderOutput.binary_masks: rendered binary masks, unit norm 3D vectors 
    index: index of the rendering to be extracted
    fields: index of the rendering to be extracted

    """
    out = {}
    for f in fields:
        match f:
            case "rgb":
                # inital renders are normalized to [0-1] floats
                out["rgb"] = (255*renderings.rgbs[index].permute(1,2,0)).byte().cpu().numpy()  # (B, 3, h, w) -> (h, w, 3) 
            case "depth":
                out["depth"] = renderings.depths[index,0].cpu().numpy()  # (1, 1, h, w) -> (h, w)
            case "mask":
                out["mask"] = renderings.binary_masks[index,0].cpu().numpy()  # (1, 1, h, w) -> (h, w)
            case "normals":
                out["normals"] = renderings.normals[index].permute(1,2,0).cpu().numpy()  # (1, 3, h, w) -> (h, w, 3)

    return out
