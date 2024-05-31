import os.path as osp
import sys
import subprocess
import re
import random
import time
import numpy as np
import cv2
import trimesh
import open3d as o3d
import torch
import yaml
import omegaconf
import scipy
import scipy.sparse
import math
import shutil
import PIL
from datetime import datetime
from tqdm import tqdm
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from pathlib import Path

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), "..")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from genzi.io import may_create_folder, parent_folder, write_image, read_lines


class Timer(object):
    def __init__(self):
        self.reset()

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def reset(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0


class KNNSearch(object):
    DTYPE = np.float32
    WORKERS = 4

    def __init__(self, data):
        self.data = np.asarray(data, dtype=self.DTYPE)
        self.kdtree = cKDTree(self.data)

    def query(self, kpts, k, return_dists=False):
        kpts = np.asarray(kpts, dtype=self.DTYPE)
        nndists, nnindices = self.kdtree.query(kpts, k=k, workers=self.WORKERS)
        if return_dists:
            return nnindices, nndists
        else:
            return nnindices

    def query_ball(self, kpt, radius):
        kpt = np.asarray(kpt, dtype=self.DTYPE)
        assert kpt.ndim == 1
        nnindices = self.kdtree.query_ball_point(kpt, radius, workers=self.WORKERS)
        return nnindices


class OptimLogger(object):

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.streams = list()
        self.spaths = list()
        may_create_folder(out_dir)

    def start_stream(self, prefix, frame_size=(512, 512), fps=10):
        spath = osp.join(self.out_dir, prefix + ".mp4")
        may_create_folder(parent_folder(spath))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        stream = cv2.VideoWriter(spath, fourcc, fps, frame_size)
        stream_id = len(self.streams)
        self.streams.append(stream)
        self.spaths.append(spath)
        return stream_id

    def append_frame(self, stream_id, image):
        image = (to_numpy(image) * 255).astype(np.uint8)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)
            image = np.tile(image, (1, 1, 3))
        if image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))
        self.streams[stream_id].write(image[..., ::-1])

    def close_streams(self, save_gif=False, fps=30, scale=512):
        for sid in range(len(self.streams)):
            self.streams[sid].release()
        if save_gif:
            for spath in self.spaths:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        spath,
                        "-vf",
                        f"fps={fps},scale={scale}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                        "-loop",
                        "0",
                        spath[:-4] + ".gif",
                    ]
                )
        self.streams = list()
        self.spaths = list()

    def save_images(self, prefix, images, shape=None):
        images = (to_numpy(images) * 255).astype(np.uint8)
        if images.ndim == 2:
            images = np.expand_dims(images, axis=2)
            images = np.tile(images, (1, 1, 3))
        if images.ndim == 4:
            if images.shape[-1] != 3:
                images = np.transpose(images, (0, 2, 3, 1))
            ipaths = list()
            for i in range(images.shape[0]):
                ipath = osp.join(self.out_dir, f"{prefix}_{i:03d}.png")
                ipaths.append(ipath)
                write_image(ipath, images[i], shape=shape)
            return ipaths
        elif images.ndim == 3:
            if images.shape[-1] != 3:
                images = np.transpose(images, (1, 2, 0))
            ipath = osp.join(self.out_dir, f"{prefix}.png")
            write_image(ipath, images, shape=shape)
            return ipath
        else:
            raise RuntimeError(f"Input images have wrong shape {images.shape}")


def get_time(fmt="%y-%m-%d_%H-%M-%S.%f"):
    return datetime.now().strftime(fmt)


def get_tqdm(num_iters, desc):
    return tqdm(range(num_iters), miniters=max(int(num_iters / 100), 1), desc=desc)


def do_step(iter_idx, num_iters, step):
    if num_iters is not None:
        return (
            iter_idx == 0 or (iter_idx + 1) % step == 0 or (iter_idx + 1) == num_iters
        )
    else:
        return iter_idx == 0 or (iter_idx + 1) % step == 0


def check_grad(params):
    do_step = True
    for param in params:
        if hasattr(param, "grad") and param.grad is not None:
            if (1 - torch.isfinite(param.grad).long()).sum() > 0:
                do_step = False
                break
    return do_step


def normalize(x, axis, eps=1e-6):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    norm = np.clip(norm, a_min=eps, a_max=None)
    return x / norm


def cosine_weights(w_start, w_end, steps, output_type="np", device="cpu"):
    w_min = min(w_start, w_end)
    w_max = max(w_start, w_end)
    curr_steps = np.arange(steps)
    weights = w_min + 0.5 * (w_max - w_min) * (
        1 + np.cos(curr_steps.astype(np.float32) / steps * np.pi)
    )
    weights = weights if w_start >= w_end else weights[::-1]
    if output_type == "pt":
        return torch.as_tensor(weights).float().to(device)
    elif output_type == "np":
        return weights.astype(np.float32)
    elif output_type == "list":
        return weights.tolist()
    else:
        raise RuntimeError(f"[!] {output_type} is not supported!")


def linear_weights(w_start, w_end, steps, output_type="list", device="cpu"):
    weights = np.linspace(w_start, w_end, steps + 1)
    weights = weights[:-1]
    if output_type == "pt":
        return torch.as_tensor(weights).float().to(device)
    elif output_type == "np":
        return weights.astype(np.float32)
    elif output_type == "list":
        return weights.tolist()
    else:
        raise RuntimeError(f"[!] {output_type} is not supported!")


def get_translate_matrix(*args):
    if len(args) == 1:
        if isinstance(args[0], float):
            tx, ty, tz = args[0], args[0], args[0]
        elif isinstance(args[0], (np.ndarray, list, tuple)):
            assert len(args[0]) == 3
            tx, ty, tz = args[0][0], args[0][1], args[0][2]
        else:
            raise RuntimeError("[!] Wrong input arguments!")
    elif len(args) == 3:
        assert isinstance(args[0], float)
        tx, ty, tz = args
    else:
        raise RuntimeError("[!] Wrong input arguments!")
    res = np.identity(4, dtype=np.float32)
    res[0, 3] = tx
    res[1, 3] = ty
    res[2, 3] = tz
    return res


def get_scale_matrix(*args):
    if len(args) == 1:
        if isinstance(args[0], float):
            sx, sy, sz = args[0], args[0], args[0]
        elif isinstance(args[0], (np.ndarray, list, tuple)):
            assert len(args[0]) == 3
            sx, sy, sz = args[0][0], args[0][1], args[0][2]
        else:
            raise RuntimeError("[!] Wrong input arguments!")
    elif len(args) == 3:
        assert isinstance(args[0], float)
        sx, sy, sz = args
    else:
        raise RuntimeError("[!] Wrong input arguments!")
    res = np.identity(4, dtype=np.float32)
    res[0, 0] = sx
    res[1, 1] = sy
    res[2, 2] = sz
    return res


def get_rotation_matrix(axis, theta):
    theta = theta * math.pi / 180.0
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    if axis == "x":
        rot = np.asarray(
            [
                [1, 0, 0],
                [0, costheta, -sintheta],
                [0, sintheta, costheta],
            ]
        )
    elif axis == "y":
        rot = np.asarray(
            [
                [costheta, 0, sintheta],
                [0, 1, 0],
                [-sintheta, 0, costheta],
            ]
        )
    elif axis == "z":
        rot = np.asarray(
            [
                [costheta, -sintheta, 0],
                [sintheta, costheta, 0],
                [0, 0, 1],
            ]
        )
    else:
        raise RuntimeError(f"[!] axis {axis} is not supported!")
    res = np.identity(4, dtype=np.float32)
    res[:3, :3] = rot
    return res


def get_rotation_matrix_between_vectors(vec1, vec2):
    is_np = isinstance(vec1, np.ndarray) or isinstance(vec2, np.ndarray)
    if is_np:
        vec1 = to_torch(vec1)
        vec2 = to_torch(vec2)
    a = vec1 / torch.linalg.norm(vec1)
    b = vec2 / torch.linalg.norm(vec2)
    v = torch.cross(a, b)
    if torch.any(v).item():
        c = torch.dot(a, b)
        s = torch.linalg.norm(v)
        kmat = torch.as_tensor(
            [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]
        ).to(v)
        res = torch.eye(3).to(v) + kmat + kmat @ kmat * ((1 - c) / (s**2))
    else:
        res = torch.eye(3).to(v)
    if is_np:
        res = to_numpy(res)
    return res


def transform_points3d(vertices, trans):
    ones = np.ones(vertices.shape[:-1] + (1,), dtype=vertices.dtype)
    vertices_h = np.concatenate((vertices, ones), axis=-1)
    vertices_h = vertices_h @ np.transpose(trans)
    return vertices_h[..., :3]


def to_trimesh(V, F, VC=None, VN=None):
    return trimesh.Trimesh(
        vertices=V,
        faces=F,
        vertex_colors=VC,
        vertex_normals=VN,
        process=False,
        validate=False,
    )


def to_o3d_mesh(V, F, VC=None):
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.copy(V)), o3d.utility.Vector3iVector(np.copy(F))
    )
    if VC is not None:
        m.vertex_colors = o3d.utility.Vector3dVector(np.copy(VC))
    return m


def to_o3d_pcd(V, VN=None, VC=None):
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.copy(V)))
    if VN is not None:
        p.normals = o3d.utility.Vector3dVector(np.copy(VN))
    if VC is not None:
        p.colors = o3d.utility.Vector3dVector(np.copy(VC))
    return p


def load_trimesh(filepath, force=None):
    return trimesh.load(filepath, force=force, process=False, validate=False)


def read_mesh(filepath, has_vc=False, rot_axes="", rot_angles=[]):
    m = o3d.io.read_triangle_mesh(filepath)
    if rot_axes is not None and rot_axes != "":
        assert len(rot_axes) == len(rot_angles)
        trans = np.identity(4, dtype=np.float32)
        for idx, axis in enumerate(rot_axes):
            trans = get_rotation_matrix(axis, rot_angles[idx]) @ trans
        m.transform(trans)
    V = np.asarray(m.vertices).astype(np.float32)
    F = np.asarray(m.triangles).astype(np.int32)
    if has_vc:
        VC = np.asarray(m.vertex_colors).astype(np.float32)
        return V, F, VC
    else:
        return V, F


def read_pcd(filepath):
    m = o3d.io.read_point_cloud(filepath)
    V = np.asarray(m.points).astype(np.float32)
    return V


def save_mesh(filepath, V, F, VC=None):
    m = to_o3d_mesh(V, F, VC)
    may_create_folder(parent_folder(filepath))
    return o3d.io.write_triangle_mesh(filepath, m)


def save_pcd(filepath, V, VN=None, VC=None):
    p = to_o3d_pcd(V, VN, VC)
    may_create_folder(parent_folder(filepath))
    return o3d.io.write_point_cloud(filepath, p)


def save_smplx_mesh(
    filepath,
    template_path,
    texture_path,
    vertices,
    Ns=160,
    Ka=1.0,
    Ks=0.5,
    Ke=0.0,
    Ni=1.45,
    d=1.0,
    illum=2,
):
    root_dir = parent_folder(filepath)
    filename = Path(filepath).stem

    lines = list()
    lines.append(f"mtllib {filename}.mtl")
    lines.append("o SMPLX-mesh")
    for i in range(len(vertices)):
        lines.append(
            f"v {vertices[i, 0]:.6f} {vertices[i, 1]:.6f} {vertices[i, 2]:.6f}"
        )
    temp_lines = read_lines(template_path)
    with open(filepath, "w") as fh:
        for line in lines + temp_lines:
            fh.write(line + "\n")

    lines = "newmtl material_0\n"
    lines += f"Ns {Ns}\n"
    lines += f"Ka {Ka} {Ka} {Ka}\n"
    lines += f"Ks {Ks} {Ks} {Ks}\n"
    lines += f"Ke {Ke} {Ke} {Ke}\n"
    lines += f"Ni {Ni} {Ni} {Ni}\n"
    lines += f"d {d}\n"
    lines += f"illum {illum}\n"
    if valid_str(texture_path):
        lines += f"map_Kd {Path(texture_path).name}\n"
    with open(osp.join(root_dir, f"{filename}.mtl"), "w") as fh:
        fh.write(lines)

    if valid_str(texture_path):
        shutil.copy(texture_path, osp.join(root_dir, Path(texture_path).name))


def generate_skeletion_mesh(points, indices, radius=0.025):
    mesh = None
    for i in range(len(indices)):
        p0 = points[indices[i, 0]]
        p1 = points[indices[i, 1]]
        c = (p0 + p1) / 2
        m = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=np.linalg.norm(p1 - p0)
        )
        r = np.identity(4, dtype=np.float32)
        r[:3, :3] = get_rotation_matrix_between_vectors(
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32), p1 - p0
        )
        t = get_translate_matrix(c)
        m.transform(t @ r)
        if mesh is None:
            mesh = m
        else:
            mesh += m
    V = np.asarray(mesh.vertices).astype(np.float32)
    F = np.asarray(mesh.triangles).astype(np.int32)
    return V, F


def fibonacci_sphere(samples=100):
    points = []
    phi = math.pi * (math.sqrt(5.0) - 1.0)
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2
        radius = math.sqrt(1 - y * y)
        theta = phi * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
    return np.asarray(points, dtype=np.float32)


def point2plane_distance(points, plane):
    assert points.ndim == 2 and plane.ndim == 1
    ones = np.ones_like(points[:, :1])
    points = np.concatenate((points, ones), axis=1)
    plane = np.expand_dims(plane, axis=0)
    dists = np.abs(np.sum(points * plane, axis=1)) / np.linalg.norm(
        plane, axis=1, keepdims=False
    )
    return dists


def generate_viewpoints(
    renderer,
    scene_mesh,
    obj_faces,
    at,
    at_normal,
    up,
    fov,
    num_viewpoints,
    distance,
    max_views,
    up_min_angle=math.pi / 3,
    up_max_angle=math.pi / 2,
    at_max_angle=math.pi * 3 / 4,
    min_ratio=0.4,
    random_when_fail=False,
    log_dir=None,
):
    obj_mesh = scene_mesh.submesh(faces_sequence=[obj_faces], append=True)
    obj_area = obj_mesh.area

    if at is None:
        at = obj_mesh.centroid
    if up is None:
        up = np.asarray([0, 0, 1], dtype=np.float32)
    up = up / np.linalg.norm(up)

    sfaces = scene_mesh.faces
    sface_flags = np.zeros((len(sfaces),), dtype=np.int32)
    sface_flags[obj_faces] = 1
    sfaces = np.concatenate((sfaces[sface_flags < 1], sfaces[sface_flags >= 1]), axis=0)
    num_sfaces = np.sum(sface_flags < 1)
    scene_mesh = to_trimesh(scene_mesh.vertices, sfaces, None, None)

    sface_areas = scene_mesh.area_faces
    sface_centers = np.mean(scene_mesh.vertices[scene_mesh.faces, :], axis=1).astype(
        np.float32
    )
    sface_normals = scene_mesh.face_normals

    unit_viewpoints = fibonacci_sphere(num_viewpoints)
    unit_viewpoints = unit_viewpoints / np.linalg.norm(
        unit_viewpoints, axis=1, keepdims=True
    )

    angles = np.arccos(
        np.clip(np.sum(unit_viewpoints * np.expand_dims(up, axis=0), axis=1), -1, 1)
    )
    unit_viewpoints = unit_viewpoints[
        np.logical_and(angles > up_min_angle, angles < up_max_angle), :
    ]
    if at_normal is not None:
        at_normal = at_normal / np.linalg.norm(at_normal)
        angles = np.arccos(
            np.clip(
                np.sum(unit_viewpoints * np.expand_dims(at_normal, axis=0), axis=1),
                -1,
                1,
            )
        )
        unit_viewpoints = unit_viewpoints[angles < at_max_angle, :]

    viewpoints = unit_viewpoints * distance + np.expand_dims(at, axis=0)

    renderer.save_current_state()
    renderer.set_cameras(eyes=viewpoints, at=at, up=up, fov=fov)

    sorast = renderer.rasterize(
        vertices=to_cuda(to_torch(scene_mesh.vertices).float()),
        faces=to_cuda(to_torch(scene_mesh.faces).int()),
        image_size=512,
    )
    sorast = sorast.detach().cpu().numpy()
    soindices = sorast[..., 3].astype(np.int32) - 1
    omasks = soindices >= num_sfaces

    vis_ratios = list()
    for vid in range(len(sorast)):
        oface_indices = soindices[vid].flatten()[omasks[vid].flatten()]
        if oface_indices.size == 0:
            vis_ratios.append(0)
        else:
            oface_indices = np.asarray(list(set(oface_indices.tolist())))

            view_dirs = viewpoints[[vid], :] - sface_centers[oface_indices, :]
            front_facing = (
                np.sum(view_dirs * sface_normals[oface_indices, :], axis=1) > 0
            )
            oface_indices = oface_indices[front_facing]
            if oface_indices.size == 0:
                vis_ratios.append(0)
            else:
                vis_ratios.append(np.sum(sface_areas[oface_indices]) / obj_area)

    vis_ratios = np.asarray(vis_ratios)
    view_rank = np.argsort(vis_ratios)[::-1]
    view_ids = view_rank[vis_ratios[view_rank] > min_ratio].tolist()
    if len(view_ids) > max_views:
        view_ids = view_ids[:max_views]
    if len(view_ids) == 0 and random_when_fail:
        view_ids = random.sample(range(len(viewpoints)), k=max_views)
    if len(view_ids) > 0:
        eyes = viewpoints[view_ids, :]
        if log_dir is not None:
            may_create_folder(log_dir)
            for idx, vidx in enumerate(view_ids):
                write_image(
                    osp.join(log_dir, f"mask{idx:03d}_dist{distance:.1f}.png"),
                    np.tile(
                        np.expand_dims(omasks[vidx].astype(np.float32), 2), (1, 1, 3)
                    ),
                )
            scolors, _, _ = renderer.render(
                tri_meshes=[scene_mesh],
                camera_ids=view_ids,
                normal_pbr=True,
                no_lighting=False,
            )
            for idx in range(len(scolors)):
                write_image(
                    osp.join(log_dir, f"view{idx:03d}_dist{distance:.1f}.png"),
                    scolors[idx],
                )
    else:
        eyes = list()
    renderer.recover_last_state()
    return eyes, at


def valid_str(x):
    return x is not None and x != ""


def clean_text(t):
    return re.sub("[^A-Za-z0-9]+", "_", t)


def join_texts(sep, texts):
    return sep.join([t.strip() for t in texts if len(t.strip()) > 0])


def hashing(arr, M):
    assert isinstance(arr, np.ndarray) and arr.ndim == 2
    N, D = arr.shape

    hash_vec = np.zeros(N, dtype=np.int64)
    for d in range(D):
        hash_vec += arr[:, d] * M**d
    return hash_vec


def omegaconf_to_dotdict(hparams):

    def _to_dot_dict(cfg):
        res = {}
        for k, v in cfg.items():
            if v is None:
                res[k] = v
            elif isinstance(v, omegaconf.DictConfig):
                res.update(
                    {k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()}
                )
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v
            elif isinstance(v, omegaconf.ListConfig):
                res[k] = omegaconf.OmegaConf.to_container(v, resolve=True)
            else:
                raise RuntimeError(
                    "[!] The type of {} is not supported.".format(type(v))
                )
        return res

    return _to_dot_dict(hparams)


def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse.FloatTensor(
        torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)
    ).coalesce()


def to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, (scipy.sparse.coo_matrix, scipy.sparse.csc_matrix)):
        return sparse_np_to_torch(x)
    elif isinstance(x, (list, tuple)):
        return [to_torch(t) for t in x]
    elif isinstance(x, dict):
        return {k: to_torch(v) for k, v in x.items()}
    elif isinstance(x, (int, float, bool)):
        return torch.as_tensor(x)
    else:
        return x


def to_cuda(x, device="cuda"):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, (list, tuple)):
        return [to_cuda(t) for t in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    else:
        return x


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return [to_numpy(t) for t in x]
    elif isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    elif isinstance(x, (int, float, bool)):
        return np.asarray(x)
    else:
        return x


def seeding(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def seeding_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def run_trainer(trainer_cls):
    cfg_cli = OmegaConf.from_cli()

    assert cfg_cli.run_mode is not None
    if cfg_cli.run_mode == "train":
        assert cfg_cli.run_cfg is not None
        cfg = OmegaConf.merge(
            OmegaConf.load(cfg_cli.run_cfg),
            cfg_cli,
        )
        cfg = omegaconf_to_dotdict(cfg)
        seeding(cfg["seed"])
        trainer = trainer_cls(cfg)
        trainer.train()
        trainer.test()
    elif cfg_cli.run_mode == "test":
        assert cfg_cli.run_ckpt is not None
        log_dir = str(Path(cfg_cli.run_ckpt).parent)
        with open(osp.join(log_dir, "config.yml"), "r") as fh:
            cfg = yaml.full_load(fh)
        cfg.update(omegaconf_to_dotdict(cfg_cli))
        cfg["test_ckpt"] = cfg_cli.run_ckpt
        seeding(cfg["seed"])
        trainer = trainer_cls(cfg)
        trainer.test()
    else:
        raise RuntimeError(f"[!] Mode {cfg_cli.run_mode} is not supported.")
