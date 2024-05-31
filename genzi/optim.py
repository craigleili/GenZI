import os.path as osp
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), "..")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


class SmplxParams(nn.Module):

    def __init__(
        self,
        smplx_model,
        transl=None,
        global_orient=None,
        use_latent_pose=True,
        use_shape_params=False,
        use_continous_rot_repr=True,
        seed=1,
    ):
        super().__init__()
        self.use_latent_pose = use_latent_pose
        self.use_shape_params = use_shape_params
        self.use_continous_rot_repr = use_continous_rot_repr
        self.seed = seed

        randg = np.random.RandomState(seed=seed)

        self.batch_size = 1
        assert smplx_model.batch_size == self.batch_size

        with torch.no_grad():
            if transl is not None:
                transl = torch.unsqueeze(transl.clone(), dim=0)
            else:
                transl = smplx_model.transl.detach().clone()
            assert transl.shape == (self.batch_size, 3)

            if global_orient is not None:
                global_orient = torch.unsqueeze(global_orient.clone(), dim=0)
            else:
                global_orient = smplx_model.global_orient.detach().clone()

            body_pose = smplx_model.body_pose.detach().clone()
            assert body_pose.shape == (self.batch_size, 21 * 3)
            if use_latent_pose:
                body_pose = torch.as_tensor(randg.randn(self.batch_size, 32)).float()
                assert body_pose.shape == (self.batch_size, 32)

            betas = smplx_model.betas.detach().clone()

            if use_continous_rot_repr:
                if global_orient.ndim == 2:
                    assert global_orient.shape == (self.batch_size, 3)
                    global_orient = axis_angle_to_matrix(global_orient)
                elif global_orient.ndim == 3:
                    assert global_orient.shape == (self.batch_size, 3, 3)
                else:
                    raise RuntimeError(
                        f"global_orient has an incorrect shape: {global_orient.shape}"
                    )
                global_orient = (
                    global_orient[..., :2].contiguous().view(self.batch_size, 3 * 2)
                )

                if not use_latent_pose:
                    body_pose = axis_angle_to_matrix(
                        body_pose.view(self.batch_size, 21, 3)
                    )
                    body_pose = (
                        body_pose[..., :2]
                        .contiguous()
                        .view(self.batch_size, 21 * 3 * 2)
                    )
            else:
                if global_orient.ndim == 2:
                    assert global_orient.shape == (self.batch_size, 3)
                elif global_orient.ndim == 3:
                    assert global_orient.shape == (self.batch_size, 3, 3)
                    global_orient = matrix_to_axis_angle(global_orient)
                else:
                    raise RuntimeError(
                        f"global_orient has an incorrect shape: {global_orient.shape}"
                    )

        self.transl = nn.Parameter(transl)
        self.global_orient = nn.Parameter(global_orient)
        self.body_pose = nn.Parameter(body_pose)
        if use_shape_params:
            self.betas = nn.Parameter(betas)
        else:
            self.register_buffer("betas", betas)

    def get_transl_params(self):
        return [self.transl]

    def get_orient_params(self):
        return [self.global_orient]

    def get_pose_params(self):
        return [self.body_pose]

    def get_shape_params(self):
        if self.use_shape_params:
            return [self.betas]
        else:
            return list()

    def get_global_params(self):
        return self.get_transl_params() + self.get_orient_params()

    def get_body_params(self):
        return self.get_pose_params() + self.get_shape_params()

    def forward(self, smplx_model, vposer):
        transl = self.transl

        global_orient = self.global_orient
        if self.use_continous_rot_repr:
            global_orient = self.from_continous_rot_repr(global_orient)
            global_orient = matrix_to_axis_angle(global_orient)
        assert global_orient.shape == (self.batch_size, 3)

        body_pose = self.body_pose
        if self.use_latent_pose:
            body_pose = (
                vposer.decode(body_pose)["pose_body"]
                .contiguous()
                .view(self.batch_size, 21 * 3)
            )
        else:
            if self.use_continous_rot_repr:
                body_pose = self.from_continous_rot_repr(
                    body_pose.view(self.batch_size * 21, 3 * 2)
                )
                body_pose = matrix_to_axis_angle(body_pose)
                body_pose = body_pose.view(self.batch_size, 21 * 3)
        assert body_pose.shape == (self.batch_size, 21 * 3)

        betas = self.betas

        smplxdict = smplx_model(
            transl=transl,
            global_orient=global_orient,
            betas=betas,
            body_pose=body_pose,
        )

        vertices = torch.squeeze(smplxdict.vertices, dim=0)
        joints = torch.squeeze(smplxdict.joints, dim=0)
        faces = smplx_model.faces_tensor.int()

        if self.use_latent_pose:
            body_pose_latent = self.body_pose
        else:
            body_pose_latent = vposer.encode(body_pose).mean

        return {
            "vertices": vertices,
            "joints": joints,
            "faces": faces,
            "transl": transl,
            "global_orient": global_orient,
            "betas": betas,
            "body_pose": body_pose,
            "body_pose_latent": body_pose_latent,
            "left_hand_pose": smplx_model.left_hand_pose,
            "right_hand_pose": smplx_model.right_hand_pose,
            "gender": smplx_model.gender,
        }

    @staticmethod
    def from_continous_rot_repr(x):
        reshaped_input = x.view(-1, 3, 2)
        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


class LearnableParams(nn.Module):

    def __init__(self, init_val=None, shape=None, dtype=torch.float32, func=None):
        super().__init__()
        self.func = func
        if init_val is not None:
            self.param = nn.Parameter(init_val).to(dtype=dtype)
        elif shape is not None:
            self.param = nn.Parameter(torch.randn(*shape)).to(dtype=dtype)
        else:
            raise RuntimeError("[!] init_val and shape cannot be both None!")

    def forward(self):
        if self.func is not None:
            return self.func(self.param)
        else:
            return self.param


class OptimWrapper(object):

    def __init__(
        self, params, lrs, optim_steps, momentum=0.9, optim_type="sgd", name=""
    ):
        self.params = params
        self.momentum = momentum
        self.optim_type = optim_type
        self.name = name
        if params is not None and len(params) > 0:
            if optim_type == "sgd":
                self.optimizer = torch.optim.SGD(params, lr=lrs[0], momentum=momentum)
            elif optim_type == "adam":
                self.optimizer = torch.optim.Adam(params, lr=lrs[0])
            elif optim_type == "adamw":
                self.optimizer = torch.optim.AdamW(params, lr=lrs[0])
            else:
                raise RuntimeError(f"{optim_type} is not supported!")

            assert len(lrs) == len(optim_steps)
            self.lrs = list()
            for idx in range(len(lrs)):
                self.lrs.extend([lrs[idx]] * optim_steps[idx])
        else:
            self.optimizer = None
            self.lrs = list()

    def zero_grad(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def step_params(self):
        if self.optimizer is not None:
            self.optimizer.step()

    def step_lr(self):
        if self.optimizer is not None:
            lr = self.lrs.pop(0)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    def get_lr(self):
        if self.optimizer is not None:
            return self.optimizer.param_groups[0]["lr"]
        else:
            return 0

    def get_name(self):
        return self.name
