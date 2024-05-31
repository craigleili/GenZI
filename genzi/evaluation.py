import os.path as osp
import sys
import numpy as np
import torch
import pickle
from omegaconf import OmegaConf
from collections import defaultdict
from pathlib import Path


ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), "..")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from genzi.io import (
    write_image,
    list_folders,
)
from genzi.misc import (
    get_time,
    join_texts,
    load_trimesh,
    omegaconf_to_dotdict,
    seeding,
    get_tqdm,
)

from genzi.metric import (
    DiversityMetric,
    PhysicalMetric,
    SemanticMetric,
)


def load_ours_results(
    exp_dir,
    scenes,
    stage,
    include_composition,
    ignore_missing=False,
):
    results = dict()
    for scene_name in scenes:
        scene_dir = osp.join(exp_dir, scene_name)
        scene_res = list()
        for interaction in list_folders(
            scene_dir, alphanum_sort=False, full_path=False
        ):
            if not include_composition and "+" in interaction:
                continue
            for stage_dir in get_stage_dirs(
                osp.join(scene_dir, interaction), stage=stage
            ):
                pkl_path = osp.join(scene_dir, interaction, stage_dir, "smplx.pkl")
                if not Path(pkl_path).is_file():
                    if ignore_missing:
                        print(f"[!] {pkl_path} is not generated!")
                        continue
                    else:
                        raise RuntimeError(f"[!] {pkl_path} is not generated!")
                with open(pkl_path, "rb") as fh:
                    smplx_dict = pickle.load(fh)
                    smplx_dict["stage"] = stage_dir
                scene_res.append(smplx_dict)
        results[scene_name] = scene_res
    return results


def get_stage_dirs(root_dir, stage):
    folders = list()
    for fname in list_folders(root_dir, "stage*", alphanum_sort=True, full_path=False):
        temp = [int(item[5:]) for item in fname.split("_") if item.startswith("stage")]
        if temp[-1] == stage:
            folders.append(fname)
    return folders


def write_csv(filepath, row_data, append=True, has_title=True):
    skip_title = True if Path(filepath).is_file() and has_title else False
    mode = "a" if append else "w"
    with open(filepath, mode) as fh:
        for row in row_data:
            if skip_title:
                skip_title = False
                continue
            fh.write(join_texts(",", [str(item) for item in row]) + "\n")


def main(cfg):
    print("[*] Start evaluation...")

    device = f'cuda:{cfg["gpus"][0]}' if torch.cuda.is_available() else "cpu"
    scenes = sorted(cfg["data.scenes"])
    exp_dir = cfg["data.exp_dir"]
    fov = cfg["data.fov"]
    eval_time = get_time()

    print(f"[*] Processing {exp_dir}")

    semantic_func = SemanticMetric(
        clip_path=cfg["vlm.clip_path"],
        renderer=None,
        image_size=cfg["render.image_size"],
        device=device,
    )

    diversity_funcs = {
        cls_num: DiversityMetric(cls_num=cls_num) for cls_num in cfg["metrics.cls_nums"]
    }

    physical_funcs = dict()
    scene_meshes = dict()
    scene_render_args = dict()
    scene_up_dirs = dict()
    for scene_name in scenes:
        scene_cfg = omegaconf_to_dotdict(
            OmegaConf.load(
                osp.join(cfg["data.root_dir"], scene_name + cfg["data.cfg_suffix"])
            )
        )
        physical_funcs[scene_name] = PhysicalMetric(
            sdf_path=scene_cfg["scene.sdf_path"]
        )
        scene_meshes[scene_name] = load_trimesh(scene_cfg["scene.mesh_path"])
        scene_render_args[scene_name] = {
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
        scene_up_dirs[scene_name] = scene_cfg["render.up_dir"]

    for stage_idx in cfg["data.stages"]:
        print(f"[*]   stage{stage_idx:03d}")

        synthesis_results = load_ours_results(
            exp_dir=exp_dir,
            scenes=scenes,
            stage=stage_idx,
            include_composition=cfg["data.include_composition"],
            ignore_missing=cfg["data.ignore_missing"],
        )

        semantic_data = defaultdict(list)
        diversity_data = defaultdict(list)
        physical_data = defaultdict(list)

        for scene_name, scene_results in synthesis_results.items():
            for iidx in get_tqdm(len(scene_results), scene_name):
                item = scene_results[iidx]
                interaction = item["prompt_id"]
                interaction_dir = osp.join(exp_dir, scene_name, interaction)
                stage_dir = osp.join(interaction_dir, item["stage"])

                prompt = join_texts(", ", [item["prompt_prefix"], item["prompt"]])

                view_data = np.load(
                    osp.join(
                        interaction_dir, f"views_stage{stage_idx:03d}", "views.npz"
                    )
                )
                viewpoints = view_data["viewpoints"]
                look_at = view_data["look_at"]

                human_mesh = load_trimesh(osp.join(stage_dir, "optim_human.obj"))

                sem_dict = semantic_func(
                    data_type=cfg["group"],
                    scene_mesh=scene_meshes[scene_name],
                    human_mesh=human_mesh,
                    viewpoints=viewpoints,
                    look_at=look_at,
                    up_dir=scene_up_dirs[scene_name],
                    fov=fov,
                    prompt=prompt,
                    render_args=scene_render_args[scene_name],
                )

                sem_mean = np.mean(sem_dict["scores"]).item()
                semantic_data[f"{scene_name}/{interaction}"].append(sem_mean)

                write_csv(
                    osp.join(
                        interaction_dir, f"stage{stage_idx:03d}_clip_{eval_time}.csv"
                    ),
                    [[item["stage"], sem_mean]],
                    append=True,
                    has_title=False,
                )

                csv_lines = [["View", "CLIP"]]
                for vidx in range(len(sem_dict["images"])):
                    write_image(
                        osp.join(stage_dir, f"tex_optim_view{vidx:03d}.png"),
                        sem_dict["images"][vidx],
                    )
                    csv_lines.append([vidx, sem_dict["scores"][vidx]])
                write_csv(
                    osp.join(stage_dir, "clip_scores.csv"),
                    csv_lines,
                    append=False,
                    has_title=False,
                )

                smplx_param = [
                    item[param_name] for param_name in cfg["metrics.smplx_params"]
                ]
                smplx_param = np.concatenate(smplx_param, axis=1)
                assert smplx_param.ndim == 2
                diversity_data[f"{scene_name}/{interaction}"].append(smplx_param)

                phy_dict = physical_funcs[scene_name](
                    vertices=item["vertices"], device=device
                )
                physical_data[f"{scene_name}/{interaction}"].append(phy_dict)

                write_csv(
                    osp.join(
                        interaction_dir,
                        f"stage{stage_idx:03d}_physical_{eval_time}.csv",
                    ),
                    [
                        ["Stage", "Contact", "NonCollision"],
                        [item["stage"], phy_dict["contact"], phy_dict["non_collision"]],
                    ],
                    append=True,
                    has_title=True,
                )

        semantic_score_all = list()
        semantic_score_max = list()
        for _, sscore in semantic_data.items():
            semantic_score_all.extend(sscore)
            semantic_score_max.append(max(sscore))
        semantic_score_meanofall = np.mean(semantic_score_all).item()
        semantic_score_meanofmax = np.mean(semantic_score_max).item()

        body_params = list()
        for _, bp in diversity_data.items():
            body_params.extend(bp)
        body_params = np.concatenate(body_params, axis=0)
        assert body_params.ndim == 2
        diversity_stats = {
            cls_num: diversity_func(body_params)
            for cls_num, diversity_func in diversity_funcs.items()
        }

        physical_stats = defaultdict(list)
        for _, item in physical_data.items():
            for cnc in item:
                physical_stats["contact"].append(cnc["contact"])
                physical_stats["non_collision"].append(cnc["non_collision"])
        for k in physical_stats.keys():
            physical_stats[k] = np.mean(physical_stats[k]).item()

        csv_lines = [
            ["Stage", "MeanOfAll", "MeanOfMax"],
            [
                f"stage{stage_idx:03d}",
                semantic_score_meanofall,
                semantic_score_meanofmax,
            ],
        ]
        write_csv(
            osp.join(exp_dir, f"clip_{eval_time}.csv"),
            csv_lines,
            append=True,
            has_title=True,
        )

        csv_lines = [["Stage"], [f"stage{stage_idx:03d}"]]
        for cls_num in sorted(diversity_stats.keys()):
            csv_lines[0].extend([f"Entropy_{cls_num}", f"ClusterSize_{cls_num}"])
            csv_lines[1].extend(
                [
                    diversity_stats[cls_num]["entropy"],
                    diversity_stats[cls_num]["mean_dist"],
                ]
            )
        write_csv(
            osp.join(exp_dir, f"diversity_{eval_time}.csv"),
            csv_lines,
            append=True,
            has_title=True,
        )

        csv_lines = [
            ["Stage", "Contact", "NonCollision"],
            [
                f"stage{stage_idx:03d}",
                physical_stats["contact"],
                physical_stats["non_collision"],
            ],
        ]
        write_csv(
            osp.join(exp_dir, f"physical_{eval_time}.csv"),
            csv_lines,
            append=True,
            has_title=True,
        )


if __name__ == "__main__":
    cfg_cli = OmegaConf.from_cli()
    assert cfg_cli.run_cfg is not None
    cfg = OmegaConf.merge(
        OmegaConf.load(cfg_cli.run_cfg),
        cfg_cli,
    )
    cfg = omegaconf_to_dotdict(cfg)
    seeding(cfg["seed"])

    main(cfg)
