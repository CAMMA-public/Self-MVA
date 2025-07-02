import torch
import torch.nn as nn
import torchvision
import timm
import math
from prompt_encoder import PromptEncoder
from typing import Optional, Tuple
from copy import deepcopy


class Multi_View_Predictor_prompt(nn.Module):
    """
    Prompt encoder initialized with SAM's weights
    """
    def __init__(self, num_cam, mode='box', init_prompt=True):
        super(Multi_View_Predictor_prompt, self).__init__()
        prompt_embed_dim = 256
        self.image_size = 1024
        vit_patch_size = 16
        image_embedding_size = self.image_size // vit_patch_size
        self.prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        if init_prompt:
            with open('weights/prompt_vit_h_4b8939.pth', "rb") as f:
                state_dict = torch.load(f)
            self.prompt_encoder.load_state_dict(state_dict)
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False

        self.num_cam = num_cam
        self.z_cam_token = nn.Parameter(torch.zeros(self.num_cam, 1, prompt_embed_dim))
        timm.layers.trunc_normal_(self.z_cam_token)

        self.pos_encoder = Encoder(prompt_embed_dim * 3, 'cuda')
        self.mode = mode
        if self.mode == 'box':
            pred_dim = 4 * self.num_cam
        else:
            pred_dim = 2 * self.num_cam
        self.observers = nn.Linear(prompt_embed_dim * 3, pred_dim)

        self.sigmoid = nn.Sigmoid()
    
    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
    
    def apply_coords_torch(
        self, coords: torch.Tensor, 
        original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.image_size
        )
        # coords = deepcopy(coords).to(torch.float)
        coords = coords.clone().to(torch.float)
        coords[..., 0] = coords[..., 0] * new_w
        coords[..., 1] = coords[..., 1] * new_h
        return coords
    
    def predict_bbox(self, x):
        device = x.device
        bs = len(x)

        # preds = self.observers(self.to_3d(x[:,-1])).view(bs, self.num_cam, 2)
        if self.mode == 'box':
            preds = self.observers(x).view(bs, self.num_cam, 4)
            # preds = self.observers(self.pos_decoder(x)).view(bs, self.num_cam, 4)
            center_x = preds[:, :, 0]
            center_y = preds[:, :, 1]
            w = self.sigmoid(preds[:, :, 2])
            h = self.sigmoid(preds[:, :, 3])
            out_x1 = center_x - w * 0.5
            out_x2 = center_x + w * 0.5
            out_y1 = center_y - h * 0.5
            out_y2 = center_y + h * 0.5
            out_prompts = torch.stack([out_x1, out_y1, out_x2, out_y2], dim=-1) # shape: [bs, num_cam, 4]
        else:
            preds = self.observers(x).view(bs, self.num_cam, 2)
            x = preds[:, :, 0]
            y = preds[:, :, 1]
            out_prompts = torch.stack([x, y], dim=-1) # shape: [bs, num_cam, 2]

        return out_prompts

    def forward(
        self, 
        cam_id, 
        original_size: Tuple[int, ...], 
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        ):
        if point_coords is not None:
            # point_coords.shape: [bs, N, 2]
            point_coords = self.apply_coords_torch(point_coords.unsqueeze(1), original_size)
            points = (point_coords, point_labels)
        else:
            points = None
        if boxes is not None:
            boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
            boxes = boxes.reshape(-1, 4) # shape: [bs, 4]
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )
        # sparse_embeddings.shape: [num_bbox, 2, 256]
        # z_cam_token.shape: [num_bbox, 1, 256]
        cam_batch = torch.ones(len(sparse_embeddings), dtype=torch.int64, device=sparse_embeddings.device) * cam_id[0]
        z_cam_token = torch.index_select(self.z_cam_token, 0, cam_batch)

        x = torch.cat([sparse_embeddings, z_cam_token], dim=1)
        x = x.reshape(len(sparse_embeddings), -1)
        out = self.pos_encoder(x)

        out_prompts = self.predict_bbox(out)

        return out, out_prompts
    
    def encode_decode(self, view_id, original_size, prompts):
        if prompts.shape[1] == 4:
            fea, preds = self.forward(cam_id=view_id, original_size=original_size, boxes=prompts)
        else:
            prompts_labels = torch.ones((len(prompts), 1), dtype=torch.int, device=prompts.device)
            fea, preds = self.forward(cam_id=view_id, original_size=original_size, point_coords=prompts, point_labels=prompts_labels)
        return fea, preds


class Encoder(nn.Module):
    def __init__(self, in_dim, device):
        super(Encoder, self).__init__()
        self.layer1 = nn.Linear(in_dim, in_dim)
        self.layer2 = nn.Linear(in_dim, in_dim)
        self.layer3 = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU()
        self.to(device)
    
    def layer_norm(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        eps = 1e-6
        return (x - mean) / (std + eps)

    def forward(self, x):
        x = self.layer1(x) + x
        x = self.layer_norm(x)
        x = self.relu(x)
        x = self.layer2(x) + x
        x = self.layer_norm(x)
        x = self.relu(x)
        x = self.layer3(x) + x
        x = self.layer_norm(x)
        x = self.relu(x)
        return x