import os.path as osp
import sys
import math
import numpy as np
import trimesh
import torch
import pyrender
import nvdiffrast.torch as dr
from copy import deepcopy

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), "..")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from genzi.misc import normalize, to_trimesh


class Renderer(object):

    def __init__(self, image_size, **kwargs):
        self.image_size = image_size
        for k, w in kwargs.items():
            setattr(self, k, w)

        self.renderer = None
        self.glctx = None

        self.cache = list()
        self._reset_cameras()

    def _reset_cameras(self):
        self.modelviews = list()
        self.camera_args = dict()

    def _get_mesh_nodes(self, tri_mesh):
        nodes = list()
        if isinstance(tri_mesh, trimesh.Trimesh):
            mesh = pyrender.Mesh.from_trimesh(tri_mesh)
            mesh_node = pyrender.Node(
                mesh=mesh, matrix=np.identity(4, dtype=np.float32)
            )
            nodes.append(mesh_node)
        elif isinstance(tri_mesh, trimesh.Scene):
            geometries = {
                name: pyrender.Mesh.from_trimesh(geom)
                for name, geom in tri_mesh.geometry.items()
            }
            for node in tri_mesh.graph.nodes_geometry:
                pose, geom_name = tri_mesh.graph[node]
                mesh_node = pyrender.Node(mesh=geometries[geom_name], matrix=pose)
                nodes.append(mesh_node)
        else:
            raise RuntimeError("[!] input tri_mesh has wrong type!")
        return nodes

    def set_cameras(self, eyes, at, up, fov, znear=0.1, zfar=20.0):
        self._reset_cameras()

        assert eyes.ndim == 2
        for i in range(eyes.shape[0]):
            eye = eyes[i]
            z_axis = normalize(eye - at, axis=0)
            x_axis = normalize(np.cross(up, z_axis), axis=0)
            y_axis = normalize(np.cross(z_axis, x_axis), axis=0)
            R = np.identity(4, dtype=np.float32)
            R[:3, :3] = np.stack((x_axis, y_axis, z_axis), axis=0).astype(np.float32)
            T = np.identity(4, dtype=np.float32)
            T[:3, 3] = -eye.astype(np.float32)
            world2cam = R @ T
            self.modelviews.append(world2cam)
        self.camera_args["yfov"] = fov * np.pi / 180
        self.camera_args["znear"] = znear
        self.camera_args["zfar"] = zfar
        self.camera_args["aspectRatio"] = 1

    def num_cameras(self):
        return len(self.modelviews)

    def save_current_state(self):
        self.cache.append((self.modelviews, self.camera_args))

    def recover_last_state(self):
        if len(self.cache) == 0:
            raise RuntimeError("[!] There is no saved state!")
        else:
            self.modelviews, self.camera_args = self.cache.pop()

    def render(
        self,
        vertices=[],
        faces=[],
        vertex_colors=[],
        tri_meshes=[],
        camera_ids=None,
        bg_color=[0.5, 0.5, 0.5, 0],
        ambient_light=[0.0, 0.0, 0.0],
        dir_light_color=[1.0, 1.0, 1.0],
        dir_light_intensity=5.0,
        pt_light_color=[1.0, 1.0, 1.0],
        pt_light_intensity=5.0,
        pt_light_position=[0.0, 0.0, 20.0],
        normal_pbr=True,
        no_lighting=False,
        all_solid=False,
        cull_faces=False,
        shadows=False,
    ):
        if camera_ids is None:
            camera_ids = list(range(self.num_cameras()))

        render_flags = pyrender.RenderFlags.RGBA
        if normal_pbr:
            render_flags |= pyrender.RenderFlags.NONE
        if no_lighting:
            render_flags |= pyrender.RenderFlags.FLAT
        if all_solid:
            render_flags |= pyrender.RenderFlags.ALL_SOLID
        if not cull_faces:
            render_flags |= pyrender.RenderFlags.SKIP_CULL_FACES
        if shadows:
            render_flags |= (
                pyrender.RenderFlags.SHADOWS_DIRECTIONAL
                | pyrender.RenderFlags.SHADOWS_SPOT
            )

        scene = pyrender.Scene(bg_color=bg_color, ambient_light=ambient_light)

        assert len(vertices) == len(faces)
        mesh_nodes = list()
        for mid in range(len(vertices)):
            V = vertices[mid]
            F = faces[mid]
            if len(vertex_colors) == len(vertices):
                VC = vertex_colors[mid]
            else:
                VC = 0.5 * np.ones_like(V)
            mesh = pyrender.Mesh.from_trimesh(to_trimesh(V, F, VC))
            mesh_node = pyrender.Node(
                mesh=mesh, matrix=np.identity(4, dtype=np.float32)
            )
            mesh_nodes.append(mesh_node)
        for tri_mesh in tri_meshes:
            mesh_nodes.extend(self._get_mesh_nodes(tri_mesh))
        for mesh_node in mesh_nodes:
            scene.add_node(mesh_node)

        if pt_light_position is not None:
            pt_light = pyrender.PointLight(
                color=pt_light_color, intensity=pt_light_intensity
            )
            pt_light_pose = np.identity(4, dtype=np.float32)
            pt_light_pose[:3, 3] = np.asarray(pt_light_position)
            pt_light_node = pyrender.Node(light=pt_light, matrix=pt_light_pose)
            scene.add_node(pt_light_node)

        dir_light = pyrender.DirectionalLight(
            color=dir_light_color, intensity=dir_light_intensity
        )
        dir_light_node = pyrender.Node(
            light=dir_light, matrix=np.identity(4, dtype=np.float32)
        )
        scene.add_node(dir_light_node)

        cam = pyrender.PerspectiveCamera(**self.camera_args)
        cam_node = pyrender.Node(camera=cam, matrix=np.identity(4, dtype=np.float32))
        scene.add_node(cam_node)

        if self.renderer is None:
            self.renderer = pyrender.OffscreenRenderer(
                viewport_width=self.image_size, viewport_height=self.image_size
            )

        colors = list()
        depths = list()
        for cid in camera_ids:
            cam_pose = np.linalg.inv(self.modelviews[cid])
            scene.set_pose(cam_node, cam_pose)
            scene.set_pose(dir_light_node, cam_pose)
            color, depth = self.renderer.render(scene=scene, flags=render_flags)
            colors.append(color)
            depths.append(depth)
        colors = np.stack(colors, axis=0).astype(np.float32) / 255
        colors, alphas = colors[..., :3], colors[..., 3]
        alphas = (alphas >= 0.5).astype(np.float32)
        depths = np.stack(depths, axis=0)
        return colors, alphas, depths

    def project(self, vertices, camera_ids=None):
        is_pt = isinstance(vertices, torch.Tensor)

        coords = self.to_ndc(vertices=vertices, camera_ids=camera_ids)
        coords = coords[..., :3] / coords[..., 3:]

        screen_x = (coords[..., 0] + 1) * 0.5 * self.image_size
        screen_y = self.image_size - (coords[..., 1] + 1) * 0.5 * self.image_size
        if is_pt:
            coords = torch.stack((screen_x, screen_y), dim=2)
        else:
            coords = np.stack((screen_x, screen_y), axis=2)
        return coords

    def to_ndc(self, vertices, camera_ids=None):
        if camera_ids is None:
            camera_ids = list(range(self.num_cameras()))
        cam = pyrender.PerspectiveCamera(**self.camera_args)
        proj = cam.get_projection_matrix()

        is_pt = isinstance(vertices, torch.Tensor)
        if is_pt:
            ones = torch.ones(
                vertices.shape[0], 1, dtype=vertices.dtype, device=vertices.device
            )
            vertices = torch.cat((vertices, ones), dim=1)
        else:
            ones = np.ones((vertices.shape[0], 1), dtype=vertices.dtype)
            vertices = np.concatenate((vertices, ones), axis=1)

        coords = list()
        for cid in camera_ids:
            mvp = proj @ self.modelviews[cid]
            if is_pt:
                mvp = torch.from_numpy(mvp).to(vertices)
                vertices_proj = vertices @ mvp.t()
            else:
                vertices_proj = vertices @ mvp.T
            coords.append(vertices_proj)
        if is_pt:
            coords = torch.stack(coords, dim=0)
        else:
            coords = np.stack(coords, axis=0)
        return coords

    @torch.no_grad()
    def rasterize(self, vertices, faces, camera_ids=None, image_size=None):
        assert isinstance(vertices, torch.Tensor) and isinstance(faces, torch.Tensor)
        if self.glctx is None:
            self.glctx = dr.RasterizeCudaContext(vertices.device)
        if image_size is None:
            image_size = self.image_size
        if camera_ids is None:
            camera_ids = list(range(self.num_cameras()))

        res = list()
        for i in camera_ids:
            coords = self.to_ndc(vertices, camera_ids=[i])
            rast_out, rast_out_isd = dr.rasterize(
                self.glctx, coords, faces, resolution=(image_size, image_size)
            )
            rast_out = torch.flip(rast_out, dims=[1])
            res.append(rast_out)
        res = torch.cat(res, dim=0)
        return res


def get_human_render_args(data_type, has_texture, render_args):

    hrender_args = deepcopy(render_args)
    if data_type == "proxs":
        if has_texture:
            hrender_args["normal_pbr"] = True
            hrender_args["no_lighting"] = False
        else:
            hrender_args["normal_pbr"] = True
            hrender_args["no_lighting"] = False
            hrender_args["dir_light_intensity"] = min(
                5, hrender_args["dir_light_intensity"]
            )
    elif data_type == "sketchfab":
        if has_texture:
            pass
        else:
            hrender_args["dir_light_intensity"] = min(
                5, hrender_args["dir_light_intensity"]
            )
    else:
        raise RuntimeError(f"[!] {data_type} is not supported!")
    return hrender_args


def render_hsi(
    data_type,
    renderer,
    render_args,
    has_human_texture,
    scene_vertices=[],
    scene_faces=[],
    scene_vertex_colors=[],
    scene_trimeshes=[],
    human_vertices=[],
    human_faces=[],
    human_vertex_colors=[],
    human_trimeshes=[],
    camera_ids=None,
):
    hrender_args = get_human_render_args(
        data_type=data_type, has_texture=has_human_texture, render_args=render_args
    )
    simages, smasks, sdepths = renderer.render(
        vertices=scene_vertices,
        faces=scene_faces,
        vertex_colors=scene_vertex_colors,
        tri_meshes=scene_trimeshes,
        camera_ids=camera_ids,
        **render_args,
    )
    himages, hmasks, hdepths = renderer.render(
        vertices=human_vertices,
        faces=human_faces,
        vertex_colors=human_vertex_colors,
        tri_meshes=human_trimeshes,
        camera_ids=camera_ids,
        **hrender_args,
    )
    overlap = smasks * hmasks
    nonoverlap = hmasks * (1 - overlap)
    depth_test = (hdepths <= sdepths).astype(np.float32)
    hsimages = np.where(
        np.expand_dims(depth_test * overlap, axis=3) >= 0.5, himages, simages
    )
    hsimages = np.where(np.expand_dims(nonoverlap, axis=3) >= 0.5, himages, hsimages)
    return hsimages
