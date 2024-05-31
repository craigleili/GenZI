import os.path as osp
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from pathlib import Path

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), "..")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from genzi.io import read_json
from genzi.misc import (
    to_numpy,
    to_torch,
    valid_str,
    load_trimesh,
    generate_viewpoints,
    KNNSearch,
)


class Scene(object):

    def __init__(self, mesh_path, sdf_path, subd_mesh_path, **kwargs):
        self.mesh_path = mesh_path
        self.sdf_path = sdf_path
        self.subd_mesh_path = subd_mesh_path
        for k, w in kwargs.items():
            setattr(self, k, w)

        self.mesh = load_trimesh(mesh_path)
        if valid_str(subd_mesh_path):
            self.subd_mesh = load_trimesh(subd_mesh_path, force="mesh")
        else:
            self.subd_mesh = self.mesh

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

    def get_name(self):
        return Path(self.mesh_path).stem

    def get_trimesh(self):
        return self.mesh

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

    def get_viewpoints(
        self,
        renderer,
        at,
        up=None,
        fov=60,
        num_viewpoints=200,
        distance=2.0,
        max_views=8,
        radius=0.25,
        use_at_normal=False,
        vpid="",
        cache_path=None,
    ):
        data = dict()

        if valid_str(cache_path) and Path(cache_path).is_file():
            with open(cache_path, "rb") as fh:
                data = pickle.load(fh)
        if vpid in data:
            return data[vpid]

        knnsearch = KNNSearch(self.subd_mesh.vertices)
        pvindices = knnsearch.query_ball(at, radius=radius)
        (pfindices,) = np.nonzero(
            np.any(np.isin(self.subd_mesh.faces, pvindices), axis=1)
        )
        (paindices,) = np.nonzero(
            np.all(np.isin(self.subd_mesh.face_adjacency, pfindices), axis=1)
        )
        cclabels = trimesh.graph.connected_component_labels(
            self.subd_mesh.face_adjacency[paindices, :],
            node_count=len(self.subd_mesh.faces),
        )
        flabels, fcounts = np.unique(cclabels, return_counts=True)
        flabels = flabels[fcounts > 1]
        pfindices = None
        pnndist = None
        for flabel in flabels.tolist():
            findices = np.nonzero(cclabels == flabel)[0]
            pmesh = self.subd_mesh.submesh(faces_sequence=[findices], append=True)
            _, pnndists = KNNSearch(pmesh.vertices).query(
                np.expand_dims(at, 0), k=1, return_dists=True
            )
            if pnndist is None or pnndists[0].item() < pnndist:
                pfindices = findices
                pnndist = pnndists[0].item()

        if use_at_normal:
            nnindices = knnsearch.query(np.expand_dims(at, 0), k=1, return_dists=False)
            assert nnindices.shape == (1,)
            at_normal = self.subd_mesh.vertex_normals[nnindices[0], :]
        else:
            at_normal = None

        eyes, at = generate_viewpoints(
            renderer=renderer,
            scene_mesh=self.subd_mesh,
            obj_faces=pfindices,
            at=at,
            at_normal=at_normal,
            up=up,
            fov=fov,
            num_viewpoints=num_viewpoints,
            distance=distance,
            max_views=max_views,
        )

        data[vpid] = (eyes, at)
        if valid_str(cache_path):
            with open(cache_path, "wb") as fh:
                pickle.dump(data, fh)
        return eyes, at
