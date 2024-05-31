import os.path as osp
import sys
import numpy as np
import scipy.cluster
import scipy.stats
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), "..")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from genzi.render import Renderer, render_hsi
from genzi.io import read_json
from genzi.misc import to_numpy, to_torch


class DiversityMetric(object):
    def __init__(self, cls_num=20):
        self.cls_num = cls_num

    def __call__(self, body_params):
        assert body_params.ndim == 2
        cls_num = self.cls_num
        if body_params.shape[0] < cls_num:
            cls_num = max(1, body_params.shape[0] // 10)
        codes, dist = scipy.cluster.vq.kmeans(body_params, cls_num)
        vecs, dist = scipy.cluster.vq.vq(body_params, codes)
        counts, bins = np.histogram(vecs, np.arange(len(codes) + 1))
        ee = scipy.stats.entropy(counts)
        return {
            "entropy": float(ee),
            "mean_dist": float(np.mean(dist)),
            "num_samples": body_params.shape[0],
        }


class PhysicalMetric(object):
    def __init__(self, sdf_path):
        if sdf_path.endswith(".json"):
            sdf_json_path = sdf_path
            sdf_npy_path = sdf_path[:-5] + "_sdf.npy"
        elif sdf_path.endswith("_sdf.npy"):
            sdf_json_path = sdf_json_path[:-8] + ".json"
            sdf_npy_path = sdf_path
        else:
            raise RuntimeError(f"[!] {sdf_path} is wrong!")
        sdf_meta = read_json(sdf_json_path)
        sdf_dim = sdf_meta["dim"]
        sdf_min = np.array(sdf_meta["min"]).astype(np.float32)
        sdf_max = np.array(sdf_meta["max"]).astype(np.float32)
        sdf = np.load(sdf_npy_path).astype(np.float32)

        self.sdf_dim = sdf_dim
        self.sdf_min = np.reshape(sdf_min, (1, 1, 3))
        self.sdf_max = np.reshape(sdf_max, (1, 1, 3))
        self.sdf = np.reshape(sdf, (1, 1, sdf_dim, sdf_dim, sdf_dim))

    def _get_torch(self, name):
        assert hasattr(self, name)
        name_pt = name + "_pt"
        if not hasattr(self, name_pt):
            setattr(self, name_pt, to_torch(getattr(self, name)))
        return getattr(self, name_pt)

    def _get_data(self, name, output_type, device):
        if output_type == "pt":
            return self._get_torch(name).to(device)
        elif output_type == "np":
            return getattr(self, name)
        else:
            raise RuntimeError(f"[!] {output_type} is not supported!")

    def get_sdf(self, vertices, device):
        is_np = isinstance(vertices, np.ndarray)
        vertices = to_torch(vertices).to(device)
        sdf_min = self._get_data("sdf_min", "pt", device)
        sdf_max = self._get_data("sdf_max", "pt", device)
        sdf_vol = self._get_data("sdf", "pt", device)

        if vertices.ndim == 2:
            is_batch = False
            vertices = torch.unsqueeze(vertices, dim=0)
        else:
            is_batch = True
        assert vertices.ndim == 3
        batch_size, num_vertices, _ = vertices.shape
        vertices = (vertices - sdf_min) / (sdf_max - sdf_min) * 2 - 1
        sdf_values = F.grid_sample(
            sdf_vol,
            vertices[..., [2, 1, 0]].view(-1, num_vertices, 1, 1, 3),
            padding_mode="border",
            align_corners=True,
        )
        sdf_values = sdf_values.reshape(batch_size, num_vertices)
        if not is_batch:
            sdf_values = torch.squeeze(sdf_values, dim=0)
        if is_np:
            return to_numpy(sdf_values)
        else:
            return sdf_values

    def __call__(self, vertices, device, contact_thresh=0):
        assert vertices.ndim == 2
        nv = len(vertices)
        x = self.get_sdf(vertices, device)
        if np.sum(x <= contact_thresh).item() < 1:
            contact_score = 0.0
        else:
            contact_score = 1.0
        non_collision_score = np.sum(x >= contact_thresh).item() / float(nv)
        return {"contact": contact_score, "non_collision": non_collision_score}


class SemanticMetric(object):
    def __init__(self, clip_path, renderer, image_size, device):
        self.clip = CLIPModel.from_pretrained(clip_path)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_path)
        self.clip.to(device)

        if renderer is not None:
            self.renderer = renderer
        else:
            self.renderer = Renderer(image_size=image_size)

        self.device = device

        self.text_embed_cache = dict()

    def get_text_embeds(self, text):
        if isinstance(text, str):
            cache_key = text
        else:
            cache_key = tuple(text)
        if cache_key in self.text_embed_cache:
            return self.text_embed_cache[cache_key]

        device = self.clip.device
        if isinstance(text, str):
            is_batch = False
        else:
            is_batch = True
        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True).to(
            device
        )
        text_embeds = self.clip.get_text_features(**inputs)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        assert text_embeds.ndim == 2
        if not is_batch:
            text_embeds = torch.squeeze(text_embeds, dim=0)
        text_embeds = to_numpy(text_embeds)

        self.text_embed_cache[cache_key] = text_embeds

        return text_embeds

    def get_image_embeds(self, images):
        device = self.clip.device
        images = to_numpy(images)
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = np.expand_dims(images, 0)
            is_batch = False
        else:
            is_batch = True
        assert images[0].ndim == 3 and images[0].shape[-1] == 3
        num_images = len(images)
        images = [
            np.asarray(images[i] * 255, dtype=np.uint8) for i in range(num_images)
        ]
        inputs = self.clip_processor(images=images, return_tensors="pt").to(device)
        image_embeds = self.clip.get_image_features(**inputs)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        assert image_embeds.ndim == 2
        if not is_batch:
            image_embeds = torch.squeeze(image_embeds, dim=0)
        return to_numpy(image_embeds)

    def get_clip_similarities(self, text_embeds, image_embeds, use_logit_scale=False):
        if text_embeds.ndim == 1:
            text_embeds = np.expand_dims(text_embeds, axis=0)
            is_text_batch = False
        else:
            is_text_batch = True
        if image_embeds.ndim == 1:
            image_embeds = np.expand_dims(image_embeds, axis=0)
            is_image_batch = False
        else:
            is_image_batch = True
        assert text_embeds.ndim == image_embeds.ndim == 2

        logits = text_embeds @ image_embeds.T
        if use_logit_scale:
            logit_scale = np.exp(to_numpy(self.clip.logit_scale))
            logits = logits * logit_scale

        if not is_image_batch:
            logits = np.squeeze(logits, 1)
        if not is_text_batch:
            logits = np.squeeze(logits, 0)
        return logits

    def __call__(
        self,
        data_type,
        scene_mesh,
        human_mesh,
        viewpoints,
        look_at,
        up_dir,
        fov,
        prompt,
        render_args,
    ):
        self.renderer.save_current_state()
        self.renderer.set_cameras(
            eyes=np.asarray(viewpoints),
            at=np.asarray(look_at),
            up=np.asarray(up_dir),
            fov=fov,
        )

        text_embeds = self.get_text_embeds(prompt)
        assert text_embeds.ndim == 1

        hsimages = render_hsi(
            data_type=data_type,
            renderer=self.renderer,
            render_args=render_args,
            has_human_texture=True,
            scene_trimeshes=[scene_mesh],
            human_trimeshes=[human_mesh],
        )
        image_embeds = self.get_image_embeds(hsimages)
        assert image_embeds.ndim == 2

        sims = self.get_clip_similarities(text_embeds, image_embeds)
        assert sims.ndim == 1 and len(sims) == len(hsimages)

        self.renderer.recover_last_state()

        return {"scores": sims, "images": hsimages}
