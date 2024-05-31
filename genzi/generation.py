import os
import os.path as osp
import sys
import wandb
import pickle
import numpy as np
import glob
import shutil
import torch
import torch.multiprocessing as mp
import smplx
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.model_loader import load_model
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image
from collections import defaultdict
from copy import deepcopy

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), "..")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from genzi.ldm_inpaint import inpainting, get_ldm_inpaint
from genzi.pose2d import (
    Pose2DPipeline,
    smplx_alphapose_torso_corrs,
    smplx_alphapose_limb_corrs,
)
from genzi.render import Renderer, render_hsi, get_human_render_args
from genzi.loss import HSILoss
from genzi.optim import SmplxParams, LearnableParams, OptimWrapper
from genzi.scene import Scene
from genzi.metric import SemanticMetric
from genzi.io import may_create_folder, write_yaml, read_lines, list_files
from genzi.misc import (
    check_grad,
    cosine_weights,
    do_step,
    generate_skeletion_mesh,
    get_rotation_matrix,
    get_time,
    get_tqdm,
    join_texts,
    omegaconf_to_dotdict,
    save_mesh,
    save_smplx_mesh,
    load_trimesh,
    seeding,
    to_numpy,
    valid_str,
    OptimLogger,
    Timer,
)


class GenZI(object):

    def __init__(self, cfg):
        self.cfg = cfg

        self.device = f'cuda:{cfg["gpus"][0]}' if torch.cuda.is_available() else "cpu"

        curr_time = get_time()
        cfg["curr_time"] = curr_time
        if not valid_str(cfg["exp_time"]):
            cfg["exp_time"] = curr_time

        wandb_dir = osp.join(cfg["log_dir"], f'{cfg["group"]}_{cfg["curr_time"]}')
        log_dir = osp.join(cfg["log_dir"], f'{cfg["group"]}_{cfg["exp_time"]}')
        cfg["log_dir"] = log_dir
        may_create_folder(wandb_dir)
        may_create_folder(log_dir)

        print("[*] Using log dir", cfg["log_dir"])

        wandb.init(
            project=cfg["project"],
            dir=wandb_dir,
            group=cfg["group"],
            notes=cfg["notes"],
            tags=cfg["tags"],
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.config.update(cfg)

        write_yaml(
            osp.join(
                cfg["log_dir"], f'{Path(cfg["run_cfg"]).stem}_{cfg["curr_time"]}.yml'
            ),
            cfg,
            flow_style=None,
        )

        wandb_cfg = {
            "id": wandb.run.id,
            "name": wandb.run.name,
            "group": wandb.run.group,
            "project": wandb.run.project,
            "url": wandb.run.url,
        }
        write_yaml(osp.join(wandb_dir, "wandb.yml"), wandb_cfg, flow_style=False)

        self.timers = defaultdict(Timer)

        self.vposer, _ = load_model(
            cfg["vposer.ckpt_path"],
            model_code=VPoser,
            remove_words_in_model_weights="vp_model.",
            disable_grad=True,
        )
        self.vposer.to(self.device)

        if len(cfg["gpus"]) > 1:
            self.ldm_inpaint = None
        else:
            self.ldm_inpaint = get_ldm_inpaint(cfg["vlm.ldm_inpaint_path"], self.device)

        self.pose2d = Pose2DPipeline(
            args_path=cfg["pose2d.args_path"], outputpath=cfg["log_dir"]
        )

        self.renderer = Renderer(image_size=cfg["render.image_size"])

        self.semantic_func = SemanticMetric(
            clip_path=cfg["vlm.clip_path"],
            renderer=self.renderer,
            image_size=cfg["render.image_size"],
            device=self.device,
        )

    def _inpaint_pose(
        self,
        stage_idx,
        prompt,
        negative_prompt,
        token_indices,
        simage_paths,
        hmask_paths,
        out_dir,
    ):
        cfg = self.cfg
        device = self.device
        gpus = cfg["gpus"]
        log_dir = cfg["log_dir"]
        inpaint_dir = cfg["vlm.inpaint_dir"]
        image_size = cfg["render.image_size"]
        inpaints_per_view = cfg["loss.inpaints_per_view"][stage_idx]

        for inpaint_id in range(inpaints_per_view):
            may_create_folder(osp.join(log_dir, out_dir + f"_inpaint{inpaint_id:03d}"))

        print("[*] Start inpainting...")
        if valid_str(inpaint_dir):
            inpaint_paths = list()
            for inpaint_id in range(inpaints_per_view):
                inpaint_paths.append(
                    list_files(
                        osp.join(inpaint_dir, out_dir + f"_inpaint{inpaint_id:03d}"),
                        "view*.png",
                        alphanum_sort=True,
                        full_path=True,
                    )
                )
        else:
            out_dir_template = osp.join(log_dir, out_dir + "_{}")

            attn_threshs = cosine_weights(
                cfg["vlm.attn_thresh_start"],
                cfg["vlm.attn_thresh_end"],
                cfg["vlm.dynamic_mask_stops"][stage_idx]
                - cfg["vlm.dynamic_mask_starts"][stage_idx],
                output_type="list",
            )
            inpaint_args = {
                "out_dir": out_dir_template,
                "num_inpaints_per_image": inpaints_per_view,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "token_indices": token_indices,
                "num_inference_steps": cfg["vlm.num_inference_steps"],
                "guidance_scale": cfg["vlm.guidance_scale"],
                "attn_res": cfg["vlm.attn_res"],
                "attn_threshs": attn_threshs,
                "attn_average_steps": cfg["vlm.attn_average_steps"],
                "dynamic_mask_start": cfg["vlm.dynamic_mask_starts"][stage_idx],
                "dynamic_mask_stop": cfg["vlm.dynamic_mask_stops"][stage_idx],
                "dilate_size": cfg["vlm.dilate_size"][stage_idx],
                "dilate_iterations": cfg["vlm.dilate_iterations"][stage_idx],
                "seed": cfg["seed"],
                "deterministic": cfg["vlm.deterministic"],
            }

            if len(gpus) > 1:
                step_size = len(simage_paths) // len(gpus)
                processes = list()
                for idx, gpu in enumerate(gpus):
                    sid = idx * step_size
                    eid = (
                        len(simage_paths)
                        if idx == len(gpus) - 1
                        else (idx + 1) * step_size
                    )

                    inpaint_args["ldm_inpaint_path"] = cfg["vlm.ldm_inpaint_path"]
                    inpaint_args["ldm_inpaint"] = None
                    inpaint_args["image_paths"] = simage_paths[sid:eid]
                    inpaint_args["mask_paths"] = hmask_paths[sid:eid]
                    inpaint_args["gpu_id"] = gpu

                    p = mp.Process(target=inpainting, kwargs=inpaint_args)
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
            else:
                assert self.ldm_inpaint is not None
                inpaint_args["ldm_inpaint_path"] = None
                inpaint_args["ldm_inpaint"] = self.ldm_inpaint
                inpaint_args["image_paths"] = simage_paths
                inpaint_args["mask_paths"] = hmask_paths
                inpaint_args["gpu_id"] = gpus[0]

                inpainting(**inpaint_args)
            inpaint_paths = list()
            for inpaint_id in range(inpaints_per_view):
                inpaint_paths.append(
                    list_files(
                        out_dir_template.format(f"inpaint{inpaint_id:03d}"),
                        "view*.png",
                        alphanum_sort=True,
                        full_path=True,
                    )
                )

        print("[*] Finished inpainting...")

        view_ids = [list() for _ in range(inpaints_per_view)]
        joints2d_torso = [list() for _ in range(inpaints_per_view)]
        joints2d_limb = [list() for _ in range(inpaints_per_view)]
        joints2d_torso_scores = [list() for _ in range(inpaints_per_view)]
        joints2d_limb_scores = [list() for _ in range(inpaints_per_view)]
        for inpaint_id in range(inpaints_per_view):
            for inpaint_path in inpaint_paths[inpaint_id]:
                vidx = int(Path(inpaint_path).stem[4:])
                inpaint_image = np.array(Image.open(inpaint_path).convert("RGB"))
                pose_name = osp.join(
                    out_dir + f"_inpaint{inpaint_id:03d}", f"pose2d{vidx:03d}.png"
                )

                pose2d_dict = self.pose2d(image=inpaint_image, im_name=pose_name)

                if len(pose2d_dict["result"]) > 0:
                    joints_score = pose2d_dict["result"][0]["kp_score"]
                    joints_torso_score = (
                        joints_score[smplx_alphapose_torso_corrs[:, 1], 0]
                        .float()
                        .to(device)
                    )
                    joints_limb_score = (
                        joints_score[smplx_alphapose_limb_corrs[:, 1], 0]
                        .float()
                        .to(device)
                    )
                    num_valid_joints = (
                        torch.sum(
                            joints_torso_score >= cfg["pose2d.score_thresh"]
                        ).item()
                        + torch.sum(
                            joints_limb_score >= cfg["pose2d.score_thresh"]
                        ).item()
                    )
                    if num_valid_joints >= cfg["pose2d.min_num_joints"]:
                        joints = pose2d_dict["result"][0]["keypoints"]
                        joints_torso = (
                            joints[smplx_alphapose_torso_corrs[:, 1], :]
                            .float()
                            .to(device)
                        )
                        joints_limb = (
                            joints[smplx_alphapose_limb_corrs[:, 1], :]
                            .float()
                            .to(device)
                        )
                        joints_torso = joints_torso / image_size
                        joints_limb = joints_limb / image_size

                        view_ids[inpaint_id].append(vidx)
                        joints2d_torso[inpaint_id].append(joints_torso)
                        joints2d_limb[inpaint_id].append(joints_limb)
                        joints2d_torso_scores[inpaint_id].append(joints_torso_score)
                        joints2d_limb_scores[inpaint_id].append(joints_limb_score)
                else:
                    print(f"[!] 2D pose detection failed for {pose_name}!")

            if len(joints2d_torso[inpaint_id]) > 0:
                joints2d_torso[inpaint_id] = torch.stack(
                    joints2d_torso[inpaint_id], dim=0
                )
            if len(joints2d_limb[inpaint_id]) > 0:
                joints2d_limb[inpaint_id] = torch.stack(
                    joints2d_limb[inpaint_id], dim=0
                )
            if len(joints2d_torso_scores[inpaint_id]) > 0:
                joints2d_torso_scores[inpaint_id] = torch.stack(
                    joints2d_torso_scores[inpaint_id], dim=0
                )
            if len(joints2d_limb_scores[inpaint_id]) > 0:
                joints2d_limb_scores[inpaint_id] = torch.stack(
                    joints2d_limb_scores[inpaint_id], dim=0
                )

        return (
            view_ids,
            joints2d_torso,
            joints2d_limb,
            joints2d_torso_scores,
            joints2d_limb_scores,
        )

    def _optim_pose(
        self,
        stage_idx,
        render_args,
        smplx_model,
        smplx_trans,
        look_at,
        valid_view_ids,
        joints2d_torso,
        joints2d_torso_scores,
        joints2d_limb,
        joints2d_limb_scores,
        ckpt_path,
        meta_dict,
        logger,
        out_dir,
    ):
        cfg = self.cfg
        device = self.device
        log_freq = cfg["log_freq"]
        image_size = cfg["render.image_size"]
        num_steps = sum(cfg["optim.steps"][stage_idx])

        rotation_axes = deepcopy(cfg["smplx.rotation_axes"])
        rotation_angles = deepcopy(cfg["smplx.rotation_angles"])

        joints2d_torso = joints2d_torso.to(device)
        joints2d_torso_scores = joints2d_torso_scores.to(device)
        joints2d_limb = joints2d_limb.to(device)
        joints2d_limb_scores = joints2d_limb_scores.to(device)

        kintree = to_numpy(smplx_model.parents)
        kintree = np.stack((kintree, np.arange(len(kintree))), axis=1)
        assert kintree.ndim == 2 and kintree.shape[1] == 2

        loss_fn = HSILoss(cfg=cfg, stage_idx=stage_idx)
        loss_fn.to(device=device)

        smplx_trans = torch.as_tensor(smplx_trans).float().to(device)
        smplx_rots = np.identity(4, dtype=np.float32)
        for idx, axis in enumerate(rotation_axes):
            smplx_rots = get_rotation_matrix(axis, rotation_angles[idx]) @ smplx_rots
        smplx_rots = torch.as_tensor(smplx_rots[:3, :3]).float().to(device)

        smplx_params = SmplxParams(
            smplx_model,
            transl=smplx_trans,
            global_orient=smplx_rots,
            use_latent_pose=cfg["smplx.use_latent_pose"],
            use_shape_params=cfg["smplx.use_shape_params"],
            use_continous_rot_repr=cfg["smplx.use_continous_rot_repr"],
        )
        smplx_params.to(device)

        if Path(ckpt_path).is_file():
            smplx_params.load_state_dict(torch.load(ckpt_path))
            with torch.no_grad():
                joints3d_init = smplx_params(smplx_model, self.vposer)[
                    "joints"
                ].detach()
        else:
            joints3d_init = None

        isparams = LearnableParams(
            init_val=torch.zeros(len(valid_view_ids)).float(), func=torch.sigmoid
        )
        isparams.to(device)

        optim_args = {
            "optim_steps": cfg["optim.steps"][stage_idx],
            "optim_type": cfg["optim.type"],
        }
        optimizers = [
            OptimWrapper(
                smplx_params.get_transl_params(),
                lrs=cfg["optim.transl_lrs"][stage_idx],
                name="transl",
                **optim_args,
            ),
            OptimWrapper(
                smplx_params.get_orient_params(),
                lrs=cfg["optim.orient_lrs"][stage_idx],
                name="orient",
                **optim_args,
            ),
            OptimWrapper(
                smplx_params.get_pose_params(),
                lrs=cfg["optim.pose_lrs"][stage_idx],
                name="pose",
                **optim_args,
            ),
            OptimWrapper(
                smplx_params.get_shape_params(),
                lrs=cfg["optim.shape_lrs"][stage_idx],
                name="shape",
                **optim_args,
            ),
            OptimWrapper(
                list(isparams.parameters()),
                lrs=cfg["optim.is_lrs"][stage_idx],
                name="is",
                **optim_args,
            ),
        ]

        stream_ids = dict()
        for vidx in valid_view_ids:
            stream_ids[vidx] = logger.start_stream(
                prefix=f"optim_view{vidx:03d}", frame_size=(image_size, image_size)
            )

        all_inpaint_scores = list()

        smplx_params.train()
        isparams.train()
        for iter_idx in get_tqdm(
            num_iters=num_steps, desc=f"Optimize {len(valid_view_ids)} views"
        ):
            for optimizer in optimizers:
                optimizer.step_lr()
                optimizer.zero_grad()

            sdict = smplx_params(smplx_model, self.vposer)
            inpaint_scores = isparams()

            joints2d_torso_proj = self.renderer.project(
                vertices=sdict["joints"][smplx_alphapose_torso_corrs[:, 0], :],
                camera_ids=valid_view_ids,
            )
            joints2d_limb_proj = self.renderer.project(
                vertices=sdict["joints"][smplx_alphapose_limb_corrs[:, 0], :],
                camera_ids=valid_view_ids,
            )
            joints2d_torso_proj = joints2d_torso_proj / image_size
            joints2d_limb_proj = joints2d_limb_proj / image_size

            loss_dict = loss_fn(
                iter_idx=iter_idx,
                inpaint_scores=inpaint_scores,
                joints2d_torso=joints2d_torso,
                joints2d_torso_scores=joints2d_torso_scores,
                joints2d_torso_proj=joints2d_torso_proj,
                joints2d_limb=joints2d_limb,
                joints2d_limb_scores=joints2d_limb_scores,
                joints2d_limb_proj=joints2d_limb_proj,
                joints3d=sdict["joints"],
                joints3d_init=joints3d_init,
                body_pose=sdict["body_pose"],
                body_pose_latent=sdict["body_pose_latent"],
                betas=sdict["betas"],
                vertices=sdict["vertices"],
                faces=sdict["faces"],
                transl=sdict["transl"],
                scene3d=self.scene3d,
                look_at=look_at,
            )
            loss = loss_dict["loss"]

            loss.backward()
            if check_grad(smplx_params.parameters()):
                if cfg["optim.grad_clip"] > 0:
                    torch.nn.utils.clip_grad_value_(
                        smplx_params.parameters(), cfg["optim.grad_clip"]
                    )
                for optimizer in optimizers:
                    optimizer.step_params()
            else:
                print(f"[!] Invalid gradients, skip iteration {iter_idx}")

            if do_step(iter_idx, num_steps, log_freq):
                log_dict = {lk: lv.item() for lk, lv in loss_dict.items()}
                for optimizer in optimizers:
                    log_dict[f"lr_{optimizer.get_name()}"] = optimizer.get_lr()
                wandb.log(log_dict)

                hsimages = render_hsi(
                    data_type=cfg["group"],
                    renderer=self.renderer,
                    render_args=render_args,
                    has_human_texture=False,
                    scene_trimeshes=[self.scene3d.get_trimesh()],
                    human_vertices=[to_numpy(sdict["vertices"])],
                    human_faces=[to_numpy(sdict["faces"])],
                    camera_ids=valid_view_ids,
                )
                for hidx, vidx in enumerate(valid_view_ids):
                    logger.save_images(f"optim_view{vidx:03d}", hsimages[hidx])
                    logger.append_frame(stream_ids[vidx], hsimages[hidx])

                all_inpaint_scores.append((iter_idx, to_numpy(inpaint_scores)))

        with torch.no_grad():
            smplx_dict = smplx_params(smplx_model, self.vposer)
            smplx_dict = {**smplx_dict, **meta_dict}

        smplx_dict = to_numpy(smplx_dict)
        with open(osp.join(out_dir, "smplx.pkl"), "wb") as fh:
            pickle.dump(smplx_dict, fh)
        torch.save(smplx_params.state_dict(), osp.join(out_dir, "params.pth"))

        self._save_inpaint_scores(
            osp.join(out_dir, "inpaint_scores.csv"), all_inpaint_scores, valid_view_ids
        )

        save_mesh(
            osp.join(out_dir, "optim_human.ply"),
            smplx_dict["vertices"],
            smplx_dict["faces"],
        )
        save_smplx_mesh(
            osp.join(out_dir, "optim_human.obj"),
            cfg["smplx.uv_path"],
            cfg["smplx.tex_path"],
            smplx_dict["vertices"],
        )

        skrender_args = get_human_render_args(
            data_type=cfg["group"], has_texture=False, render_args=render_args
        )
        skvertices, skfaces = generate_skeletion_mesh(smplx_dict["joints"], kintree[1:])
        skimages, skalphas, _ = self.renderer.render(
            vertices=[skvertices],
            faces=[skfaces],
            camera_ids=valid_view_ids,
            **skrender_args,
        )
        alpha = 0.2
        for idx, vidx in enumerate(valid_view_ids):
            pose2d_path = osp.join(out_dir, f"pose2d{vidx:03d}.png")
            pose2d_image = (
                np.array(Image.open(pose2d_path).convert("RGB")).astype(np.float32)
                / 255
            )
            compos_image = np.where(
                np.expand_dims(skalphas[idx], 2),
                alpha * pose2d_image + (1 - alpha) * skimages[idx],
                pose2d_image,
            )
            Image.fromarray((compos_image * 255).astype(np.uint8)).save(
                osp.join(out_dir, f"pose2d{vidx:03d}_optim.png")
            )

        logger.close_streams()

    def _optim_prompt(
        self,
        prompt_id,
        prompt,
        prompt_prefix,
        prompt_suffix,
        negative_prompt,
        token_indices,
        viewpoints,
        look_at,
        up_dir,
        fov,
        render_args,
        scene_name,
        interaction_label,
    ):
        cfg = self.cfg
        device = self.device
        log_dir = cfg["log_dir"]
        sp_dir = osp.join(scene_name, prompt_id)

        look_at = np.asarray(look_at)
        up_dir = np.asarray(up_dir)

        gender = cfg["smplx.gender"]

        smplx_model = smplx.create(
            model_path=cfg["smplx.model_path"],
            model_type=cfg["smplx.model_type"],
            batch_size=cfg["smplx.batch_size"],
            gender=gender,
            num_pca_comps=cfg["smplx.num_pca_comps"],
        )
        smplx_model.to(device)
        smplx_model.requires_grad_(False)

        failures = list()

        prior_dirs = [""]
        for stage_idx in range(len(cfg["optim.steps"])):
            if viewpoints is None:
                sviewpoints, slook_at = self.scene3d.get_viewpoints(
                    renderer=self.renderer,
                    at=look_at,
                    up=up_dir,
                    fov=fov,
                    num_viewpoints=cfg["data.num_viewpoints"],
                    distance=cfg["data.view_distances"][stage_idx],
                    max_views=cfg["data.max_views"],
                    radius=cfg["data.patch_radius"],
                    use_at_normal=cfg["data.use_at_normal"],
                    vpid=prompt_id,
                    cache_path=None,
                )
                smplx_trans = 0.5 * (np.mean(sviewpoints, axis=0) + slook_at)
            else:
                assert isinstance(viewpoints, (list, tuple)) and len(viewpoints) == len(
                    cfg["optim.steps"]
                )
                sviewpoints, slook_at = np.asarray(viewpoints[0]), look_at
                smplx_trans = slook_at
            assert sviewpoints.ndim == 2

            self.renderer.save_current_state()
            self.renderer.set_cameras(eyes=sviewpoints, at=slook_at, up=up_dir, fov=fov)
            used_views = list(range(self.renderer.num_cameras()))

            simages, _, _ = self.renderer.render(
                tri_meshes=[self.scene3d.get_trimesh()],
                camera_ids=used_views,
                **render_args,
            )
            assert simages.ndim == 4
            simage_paths = list()
            may_create_folder(osp.join(log_dir, sp_dir, f"views_stage{stage_idx:03d}"))
            for idx, vidx in enumerate(used_views):
                simage = Image.fromarray((simages[idx] * 255).astype(np.uint8))
                simage_path = osp.join(
                    log_dir,
                    sp_dir,
                    f"views_stage{stage_idx:03d}",
                    f"view{vidx:03d}.png",
                )
                simage.save(simage_path)
                simage_paths.append(simage_path)
            np.savez(
                osp.join(log_dir, sp_dir, f"views_stage{stage_idx:03d}", "views.npz"),
                viewpoints=sviewpoints,
                look_at=slook_at,
            )

            ip_dirs = list()
            for prior_dir in prior_dirs:
                ckpt_path = osp.join(log_dir, sp_dir, prior_dir, "params.pth")
                if Path(ckpt_path).is_file():
                    smplx_params = SmplxParams(
                        smplx_model,
                        use_latent_pose=cfg["smplx.use_latent_pose"],
                        use_shape_params=cfg["smplx.use_shape_params"],
                        use_continous_rot_repr=cfg["smplx.use_continous_rot_repr"],
                    )
                    smplx_params.to(device)
                    smplx_params.load_state_dict(torch.load(ckpt_path))

                    with torch.no_grad():
                        smplx_dict = smplx_params(smplx_model, self.vposer)

                    hrender_args = get_human_render_args(
                        data_type=cfg["group"],
                        has_texture=False,
                        render_args=render_args,
                    )
                    _, halphas, _ = self.renderer.render(
                        vertices=[to_numpy(smplx_dict["vertices"])],
                        faces=[to_numpy(smplx_dict["faces"])],
                        camera_ids=used_views,
                        **hrender_args,
                    )
                    hmask_dir = osp.join(
                        log_dir,
                        sp_dir,
                        "masks_"
                        + join_texts("_", [prior_dir, f"stage{stage_idx:03d}"]),
                    )
                    may_create_folder(hmask_dir)
                    hmask_paths = list()
                    for idx, vidx in enumerate(used_views):
                        hmask_path = osp.join(hmask_dir, f"mask{vidx:03d}.png")
                        hmask_paths.append(hmask_path)
                        Image.fromarray((halphas[idx] * 255).astype(np.uint8)).save(
                            hmask_path
                        )
                else:
                    hmask_paths = [None] * len(used_views)

                (
                    view_ids,
                    joints2d_torso,
                    joints2d_limb,
                    joints2d_torso_scores,
                    joints2d_limb_scores,
                ) = self._inpaint_pose(
                    stage_idx=stage_idx,
                    prompt=join_texts(", ", [prompt_prefix, prompt, prompt_suffix]),
                    negative_prompt=join_texts(
                        ", ", [negative_prompt, self.neg_prompts]
                    ),
                    token_indices=token_indices,
                    simage_paths=simage_paths,
                    hmask_paths=hmask_paths,
                    out_dir=osp.join(
                        sp_dir, join_texts("_", [prior_dir, f"stage{stage_idx:03d}"])
                    ),
                )

                for inpaint_id in range(cfg["loss.inpaints_per_view"][stage_idx]):
                    ip_dir = join_texts(
                        "_",
                        [
                            prior_dir,
                            f"stage{stage_idx:03d}",
                            f"inpaint{inpaint_id:03d}",
                        ],
                    )

                    if len(view_ids[inpaint_id]) == 0:
                        failures.append((scene_name, prompt_id, ip_dir))
                        print("[!] No valid inpaintings for", " ".join(failures[-1]))
                        continue

                    ip_dirs.append(ip_dir)
                    out_dir = osp.join(log_dir, sp_dir, ip_dir)

                    logger = OptimLogger(out_dir)

                    meta_dict = {
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "prompt_prefix": prompt_prefix,
                        "prompt_suffix": prompt_suffix,
                        "neg_prompt": negative_prompt,
                        "scene_name": scene_name,
                        "interaction": interaction_label,
                    }

                    self._optim_pose(
                        stage_idx=stage_idx,
                        render_args=render_args,
                        smplx_model=smplx_model,
                        smplx_trans=smplx_trans,
                        look_at=slook_at,
                        valid_view_ids=view_ids[inpaint_id],
                        joints2d_torso=joints2d_torso[inpaint_id],
                        joints2d_torso_scores=joints2d_torso_scores[inpaint_id],
                        joints2d_limb=joints2d_limb[inpaint_id],
                        joints2d_limb_scores=joints2d_limb_scores[inpaint_id],
                        ckpt_path=osp.join(log_dir, sp_dir, prior_dir, "params.pth"),
                        meta_dict=meta_dict,
                        logger=logger,
                        out_dir=out_dir,
                    )

            if stage_idx == 0:
                semantic_data = list()
                for ip_dir in ip_dirs:
                    sem_dict = self.semantic_func(
                        data_type=cfg["group"],
                        scene_mesh=self.scene3d.get_trimesh(),
                        human_mesh=load_trimesh(
                            osp.join(log_dir, sp_dir, ip_dir, "optim_human.obj")
                        ),
                        viewpoints=sviewpoints,
                        look_at=slook_at,
                        up_dir=up_dir,
                        fov=fov,
                        prompt=join_texts(", ", [prompt_prefix, prompt]),
                        render_args=render_args,
                    )
                    semantic_data.append((ip_dir, np.mean(sem_dict["scores"]).item()))
                with open(
                    osp.join(log_dir, sp_dir, f"stage{stage_idx:03d}_ranks.csv"), "w"
                ) as fh:
                    for ip_dir, rscore in semantic_data:
                        fh.write(f"{ip_dir},{rscore}\n")
                semantic_data = sorted(semantic_data, key=lambda _x: -_x[1])
                prior_dirs = [
                    semantic_data[iidx][0]
                    for iidx in range(int(len(semantic_data) * 0.5))
                ]
            else:
                prior_dirs = ip_dirs

            self.renderer.recover_last_state()

        with open(osp.join(log_dir, f'failures_{cfg["curr_time"]}.csv'), "a+") as fh:
            for sf in failures:
                fh.write(",".join(sf) + "\n")

    def _save_inpaint_scores(self, filepath, inpaint_scores, view_ids):
        with open(filepath, "w") as fh:
            linestr = ["View"] + [
                f"Step{inpaint_scores[tidx][0]}" for tidx in range(len(inpaint_scores))
            ]
            fh.write(",".join(linestr) + "\n")
            for idx, vidx in enumerate(view_ids):
                linestr = [f"{vidx:03d}"] + [
                    f"{inpaint_scores[tidx][1][idx]:.4f}"
                    for tidx in range(len(inpaint_scores))
                ]
                fh.write(",".join(linestr) + "\n")

    def init_scene(self, yml_path):
        cfg = self.cfg
        yml_name = Path(yml_path).name

        print(f"[*] Initializing scene {yml_name}")

        scene_cfg = omegaconf_to_dotdict(OmegaConf.load(yml_path))
        write_yaml(osp.join(cfg["log_dir"], yml_name), scene_cfg, flow_style=None)
        self.scene_cfg = scene_cfg

        neg_prompts = read_lines(cfg["vlm.neg_prompt_path"])
        self.neg_prompts = join_texts(", ", neg_prompts)

        self.scene3d = Scene(
            mesh_path=scene_cfg["scene.mesh_path"],
            sdf_path=scene_cfg["scene.sdf_path"],
            subd_mesh_path=scene_cfg["scene.subd_mesh_path"],
        )

    def run_scenes(self):
        print(f"[*] Start optimization on all scenes")

        for scene_name in cfg["data.scenes"]:
            self.init_scene(
                osp.join(cfg["data.root_dir"], scene_name + cfg["data.cfg_suffix"])
            )
            scene_cfg = self.scene_cfg

            render_args = {
                "bg_color": scene_cfg["render.bg_color"],
                "ambient_light": scene_cfg["render.ambient_light"],
                "dir_light_color": scene_cfg["render.dir_light_color"],
                "dir_light_intensity": scene_cfg["render.dir_light_intensity"],
                "pt_light_color": scene_cfg["render.pt_light_color"],
                "pt_light_intensity": scene_cfg["render.pt_light_intensity"],
                "pt_light_position": scene_cfg["render.pt_light_position"],
                "normal_pbr": scene_cfg["render.normal_pbr"],
                "no_lighting": scene_cfg["render.no_lighting"],
                "all_solid": scene_cfg["render.all_solid"],
                "cull_faces": scene_cfg["render.cull_faces"],
                "shadows": scene_cfg["render.shadows"],
            }

            for pid in range(len(scene_cfg["prompts"])):
                print(
                    f'[*] Scene: {scene_name}; Prompt: {scene_cfg["prompt_ids"][pid]}'
                )
                self._optim_prompt(
                    prompt_id=scene_cfg["prompt_ids"][pid],
                    prompt=scene_cfg["prompts"][pid],
                    prompt_prefix=scene_cfg["prompt_prefix"],
                    prompt_suffix=scene_cfg["prompt_suffix"],
                    negative_prompt=scene_cfg["neg_prompts"][pid],
                    token_indices=scene_cfg["token_indices"],
                    viewpoints=scene_cfg["viewpoints"][pid],
                    look_at=scene_cfg["lookats"][pid],
                    up_dir=scene_cfg["render.up_dir"],
                    fov=cfg["data.fov"],
                    render_args=render_args,
                    scene_name=scene_name,
                    interaction_label=scene_cfg["interactions"][pid],
                )
            wandb.log({f"time.{tn}": tt.average_time for tn, tt in self.timers.items()})

        print(f"[*] Finished optimization on all scenes")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    cfg_cli = OmegaConf.from_cli()
    assert cfg_cli.run_cfg is not None
    cfg = OmegaConf.merge(
        OmegaConf.load(cfg_cli.run_cfg),
        cfg_cli,
    )
    cfg = omegaconf_to_dotdict(cfg)
    seeding(cfg["seed"])
    app = GenZI(cfg)
    app.run_scenes()
