import os.path as osp
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mesh_intersection.bvh_search_tree import BVH
from mesh_intersection.loss import DistanceFieldPenetrationLoss

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), "..")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from genzi.misc import linear_weights


class HSILoss(nn.Module):

    def __init__(self, cfg, stage_idx):
        super().__init__()
        self.cfg = cfg
        self.stage_idx = stage_idx

        self.joint2d_rho = cfg["loss.joint2d_rho"]
        self.joint3d_rho = cfg["loss.joint3d_rho"]
        self.inpaint_min_views = cfg["loss.inpaint_min_views"]
        self.beta0_weight = cfg["loss.beta0_weight"]
        self.scene_intersect_thresh = cfg["loss.scene_intersect_thresh"]

        self.bvh = BVH(max_collisions=8)
        self.dfp_loss = DistanceFieldPenetrationLoss(
            sigma=0.001, point2plane=False, vectorized=True, penalize_outside=True
        )

        self.angle_loss = SMPLXAnglePrior()

        optim_steps = cfg["optim.steps"][stage_idx]
        is_weights = cfg["loss.inpaint_score_weights"][stage_idx]
        jt_weights = cfg["loss.joint2d_torso_weights"][stage_idx]
        jl_weights = cfg["loss.joint2d_limb_weights"][stage_idx]
        vp_weights = cfg["loss.vposer_weights"][stage_idx]
        bt_weights = cfg["loss.beta_weights"][stage_idx]
        si_weights = cfg["loss.scene_intersect_weights"][stage_idx]
        nc_weights = cfg["loss.scene_nocontact_weights"][stage_idx]
        sp_weights = cfg["loss.self_intersect_weights"][stage_idx]
        ap_weights = cfg["loss.angle_weights"][stage_idx]
        ft_weights = cfg["loss.floating_weights"][stage_idx]
        j3_weights = cfg["loss.joint3d_weights"][stage_idx]

        self.inpaint_score_weights = list()
        self.joint2d_torso_weights = list()
        self.joint2d_limb_weights = list()
        self.vposer_weights = list()
        self.beta_weights = list()
        self.scene_intersect_weights = list()
        self.scene_nocontact_weights = list()
        self.self_intersect_weights = list()
        self.angle_weights = list()
        self.floating_weights = list()
        self.joint3d_weights = list()
        for idx in range(len(optim_steps)):
            osteps = optim_steps[idx]
            self.inpaint_score_weights.extend(
                linear_weights(is_weights[idx], is_weights[idx + 1], osteps)
            )
            self.joint2d_torso_weights.extend(
                linear_weights(jt_weights[idx], jt_weights[idx + 1], osteps)
            )
            self.joint2d_limb_weights.extend(
                linear_weights(jl_weights[idx], jl_weights[idx + 1], osteps)
            )
            self.vposer_weights.extend(
                linear_weights(vp_weights[idx], vp_weights[idx + 1], osteps)
            )
            self.beta_weights.extend(
                linear_weights(bt_weights[idx], bt_weights[idx + 1], osteps)
            )
            self.scene_intersect_weights.extend(
                linear_weights(si_weights[idx], si_weights[idx + 1], osteps)
            )
            self.scene_nocontact_weights.extend(
                linear_weights(nc_weights[idx], nc_weights[idx + 1], osteps)
            )
            self.self_intersect_weights.extend(
                linear_weights(sp_weights[idx], sp_weights[idx + 1], osteps)
            )
            self.angle_weights.extend(
                linear_weights(ap_weights[idx], ap_weights[idx + 1], osteps)
            )
            self.floating_weights.extend(
                linear_weights(ft_weights[idx], ft_weights[idx + 1], osteps)
            )
            self.joint3d_weights.extend(
                linear_weights(j3_weights[idx], j3_weights[idx + 1], osteps)
            )

        self.register_buffer("zeros", torch.as_tensor(0).float())

    def forward(
        self,
        iter_idx,
        inpaint_scores,
        joints2d_torso,
        joints2d_torso_scores,
        joints2d_torso_proj,
        joints2d_limb,
        joints2d_limb_scores,
        joints2d_limb_proj,
        joints3d,
        joints3d_init,
        body_pose,
        body_pose_latent,
        betas,
        vertices,
        faces,
        transl,
        scene3d,
        look_at,
    ):

        loss = 0

        if (
            len(inpaint_scores) >= self.inpaint_min_views
            and self.inpaint_score_weights[iter_idx] > 0
        ):
            loss_inpaint_scores = torch.nn.functional.relu(
                self.inpaint_min_views - torch.sum(inpaint_scores)
            )
            loss = loss + self.inpaint_score_weights[iter_idx] * loss_inpaint_scores
        else:
            loss_inpaint_scores = self.zeros

        if self.joint2d_torso_weights[iter_idx] > 0:
            loss_joint2d_torso = torch.sum(
                torch.sum(
                    GMoF(joints2d_torso - joints2d_torso_proj, rho=self.joint2d_rho),
                    dim=-1,
                )
                * joints2d_torso_scores,
                dim=-1,
            )
            loss_joint2d_torso = torch.sum(
                loss_joint2d_torso * (inpaint_scores / torch.sum(inpaint_scores))
            )
            loss = loss + self.joint2d_torso_weights[iter_idx] * loss_joint2d_torso
        else:
            loss_joint2d_torso = self.zeros

        if self.joint2d_limb_weights[iter_idx] > 0:
            loss_joint2d_limb = torch.sum(
                torch.sum(
                    GMoF(joints2d_limb - joints2d_limb_proj, rho=self.joint2d_rho),
                    dim=-1,
                )
                * joints2d_limb_scores,
                dim=-1,
            )
            loss_joint2d_limb = torch.sum(
                loss_joint2d_limb * (inpaint_scores / torch.sum(inpaint_scores))
            )
            loss = loss + self.joint2d_limb_weights[iter_idx] * loss_joint2d_limb
        else:
            loss_joint2d_limb = self.zeros

        if self.vposer_weights[iter_idx] > 0:
            loss_vposer = torch.mean(body_pose_latent**2)
            loss = loss + self.vposer_weights[iter_idx] * loss_vposer
        else:
            loss_vposer = self.zeros

        if self.beta_weights[iter_idx] > 0:
            bw = torch.ones_like(betas[:1])
            bw[0, 0] = self.beta0_weight
            loss_betas = torch.mean((bw * betas) ** 2)
            loss = loss + self.beta_weights[iter_idx] * loss_betas
        else:
            loss_betas = self.zeros

        if self.scene_intersect_weights[iter_idx] > 0:
            sdf_values = scene3d.get_sdf(vertices, vertices.device)
            sdf_flags = sdf_values < self.scene_intersect_thresh
            if sdf_flags.sum().item() < 1:
                loss_scene_intersect = self.zeros
            else:
                loss_scene_intersect = torch.mean(
                    torch.abs(sdf_values[sdf_flags] - self.scene_intersect_thresh)
                )
            loss = loss + self.scene_intersect_weights[iter_idx] * loss_scene_intersect
        else:
            loss_scene_intersect = self.zeros

        if self.scene_nocontact_weights[iter_idx] > 0:
            assert sdf_values is not None and sdf_flags is not None
            if sdf_flags.sum().item() < 1:
                loss_scene_nocontact = (
                    torch.min(sdf_values) - self.scene_intersect_thresh
                )
            else:
                loss_scene_nocontact = self.zeros
            loss = loss + self.scene_nocontact_weights[iter_idx] * loss_scene_nocontact
        else:
            loss_scene_nocontact = self.zeros

        if self.self_intersect_weights[iter_idx] > 0:
            triangles = vertices[faces]
            triangles = torch.unsqueeze(triangles, dim=0)
            with torch.no_grad():
                collision_idxs = self.bvh(triangles)
            if collision_idxs.ge(0).sum().item() > 0:
                loss_self_intersect = torch.mean(
                    self.dfp_loss(triangles, collision_idxs)
                )
                loss = (
                    loss + self.self_intersect_weights[iter_idx] * loss_self_intersect
                )
            else:
                loss_self_intersect = self.zeros
        else:
            loss_self_intersect = self.zeros

        if self.angle_weights[iter_idx] > 0:
            loss_angles = self.angle_loss(body_pose, with_pelvis=False)
            loss = loss + self.angle_weights[iter_idx] * loss_angles
        else:
            loss_angles = self.zeros

        if self.floating_weights[iter_idx] > 0:
            loss_floating = torch.mean(
                torch.sigmoid(10 * (transl[:, 2] - look_at[2].item()))
            )
            loss = loss + self.floating_weights[iter_idx] * loss_floating
        else:
            loss_floating = self.zeros

        if joints3d_init is not None and self.joint3d_weights[iter_idx] > 0:
            loss_joints3d = torch.mean(
                torch.sum(GMoF(joints3d - joints3d_init, rho=self.joint3d_rho), dim=-1)
            )
            loss = loss + self.joint3d_weights[iter_idx] * loss_joints3d
        else:
            loss_joints3d = self.zeros

        return {
            "loss": loss,
            "loss_inpaint_scores": loss_inpaint_scores,
            "loss_joint2d_torso": loss_joint2d_torso,
            "loss_joint2d_limb": loss_joint2d_limb,
            "loss_vposer": loss_vposer,
            "loss_betas": loss_betas,
            "loss_scene_intersect": loss_scene_intersect,
            "loss_scene_nocontact": loss_scene_nocontact,
            "loss_self_intersect": loss_self_intersect,
            "loss_angles": loss_angles,
            "loss_floating": loss_floating,
            "loss_joints3d": loss_joints3d,
        }


class SMPLXAnglePrior(nn.Module):

    def __init__(
        self,
    ):
        super().__init__()

        clip_idxs_signs = torch.as_tensor(
            [
                (1, 0, 1),
                (2, 0, 1),
                (3, 0, -1),
                (4, 0, -1),
                (5, 0, -1),
                (6, 0, -1),
                (7, 0, -1),
                (8, 0, -1),
                (9, 0, -1),
                (12, 0, -1),
                (13, 1, 1),
                (14, 1, -1),
                (16, 1, 1),
                (17, 1, -1),
                (18, 1, 1),
                (19, 1, -1),
            ]
        ).int()

        zero_idxs = torch.as_tensor(
            [
                (10, 0),
                (10, 1),
                (10, 2),
                (11, 0),
                (11, 1),
                (11, 2),
                (15, 0),
                (15, 1),
                (15, 2),
                (20, 1),
                (21, 1),
            ]
        ).int()

        self.register_buffer("clip_idxs_signs", clip_idxs_signs)
        self.register_buffer("zero_idxs", zero_idxs)

    def forward(self, pose, with_pelvis=False):
        assert pose.ndim == 2
        cdata = self.clip_idxs_signs
        zdata = self.zero_idxs
        if not with_pelvis:
            assert pose.shape[1] == 21 * 3
            cdata = torch.clone(cdata)
            cdata[:, 0] -= 1
            zdata = torch.clone(zdata)
            zdata[:, 0] -= 1
        else:
            assert pose.shape[1] == 22 * 3

        cidxs = cdata[:, 0] * 3 + cdata[:, 1]
        csigns = cdata[:, 2]
        cres = F.relu(pose[:, cidxs] * torch.unsqueeze(csigns, 0))

        zidxs = zdata[:, 0] * 3 + zdata[:, 1]
        zres = torch.abs(pose[:, zidxs])

        loss = torch.mean(torch.cat((cres, zres), dim=1))
        return loss


def GMoF(residual, rho=0.2):
    squared_res = residual**2
    dist = torch.div(squared_res, squared_res + rho**2)
    return rho**2 * dist
