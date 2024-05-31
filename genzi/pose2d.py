import os
import os.path as osp
import sys
import platform
import time
import numpy as np
import torch
import alphapose
from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.pPose_nms import pose_nms, write_json
from alphapose.utils.presets import SimpleTransform, SimpleTransform3DSMPL
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.models import builder
from alphapose.utils.config import update_config
from detector.apis import get_detector
from alphapose.utils.vis import getTime
from pathlib import Path

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), "..")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

ALPHAPOSE_DIR = osp.join(osp.abspath(osp.dirname(alphapose.__file__)), "..")

smplx_alphapose_corrs = np.asarray(
    [
        [0, 19],
        [1, 11],
        [2, 12],
        [4, 13],
        [5, 14],
        [7, 15],
        [8, 16],
        [10, 20],
        [11, 21],
        [12, 18],
        [15, 0],
        [16, 5],
        [17, 6],
        [18, 7],
        [19, 8],
        [20, 9],
        [21, 10],
    ],
    dtype=np.int32,
)

smplx_alphapose_torso_corrs = np.asarray(
    [
        [0, 19],
        [1, 11],
        [2, 12],
        [12, 18],
        [16, 5],
        [17, 6],
    ],
    dtype=np.int32,
)

smplx_alphapose_limb_corrs = np.asarray(
    [
        [4, 13],
        [5, 14],
        [7, 15],
        [8, 16],
        [10, 20],
        [11, 21],
        [15, 0],
        [18, 7],
        [19, 8],
        [20, 9],
        [21, 10],
    ],
    dtype=np.int32,
)

# https://github.com/MVIG-SJTU/AlphaPose/blob/master/scripts/demo_api.py


class Pose2DPipeline(object):

    def __init__(self, args_path, **kwargs):
        args = update_config(args_path)
        args.cfg = osp.join(ALPHAPOSE_DIR, args.cfg)
        args.checkpoint = osp.join(ALPHAPOSE_DIR, args.checkpoint)
        exp_time = time.strftime("%y-%m-%d_%H-%M-%S")
        args.outputpath = osp.join(args.outputpath, exp_time)
        for k, w in kwargs.items():
            args[k] = w

        cfg = update_config(args.cfg)

        if platform.system() == "Windows":
            args.sp = True

        args.gpus = (
            [int(i) for i in args.gpus.split(",")]
            if torch.cuda.device_count() >= 1
            else [-1]
        )
        args.device = torch.device(
            "cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu"
        )
        args.detbatch = args.detbatch * len(args.gpus)
        args.posebatch = args.posebatch * len(args.gpus)
        args.tracking = args.pose_track or args.pose_flow or args.detector == "tracker"

        self.args = args
        self.cfg = cfg

        self.det_loader = DetectionLoader(get_detector(args), cfg, args)

        pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        print("Loading pose model from %s..." % (args.checkpoint,))
        pose_model.load_state_dict(
            torch.load(args.checkpoint, map_location=args.device)
        )
        if len(args.gpus) > 1:
            pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(
                args.device
            )
        else:
            pose_model.to(args.device)
        pose_model.eval()
        self.pose_model = pose_model

        self.pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)

    @torch.no_grad()
    def __call__(self, image, im_name="temp.png"):
        writer = DataWriter(self.cfg, self.args)

        runtime_profile = {"dt": [], "pt": [], "pn": []}
        pose = None

        start_time = getTime()

        (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = (
            self.det_loader.process(im_name, image).read()
        )
        if orig_img is None:
            raise Exception("no image is given")
        if boxes is None or boxes.nelement() == 0:
            if self.args.profile:
                ckpt_time, det_time = getTime(start_time)
                runtime_profile["dt"].append(det_time)
            writer.save(None, None, None, None, None, orig_img, im_name)
            if self.args.profile:
                ckpt_time, pose_time = getTime(ckpt_time)
                runtime_profile["pt"].append(pose_time)
            pose = writer.start()
            if self.args.profile:
                ckpt_time, post_time = getTime(ckpt_time)
                runtime_profile["pn"].append(post_time)
        else:
            if self.args.profile:
                ckpt_time, det_time = getTime(start_time)
                runtime_profile["dt"].append(det_time)
            inps = inps.to(self.args.device)
            if self.args.flip:
                inps = torch.cat((inps, flip(inps)))
            hm = self.pose_model(inps)
            if self.args.flip:
                hm_flip = flip_heatmap(
                    hm[int(len(hm) / 2) :], self.pose_dataset.joint_pairs, shift=True
                )
                hm = (hm[0 : int(len(hm) / 2)] + hm_flip) / 2
            if self.args.profile:
                ckpt_time, pose_time = getTime(ckpt_time)
                runtime_profile["pt"].append(pose_time)
            hm = hm.cpu()
            writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
            pose = writer.start()
            if self.args.profile:
                ckpt_time, post_time = getTime(ckpt_time)
                runtime_profile["pn"].append(post_time)

        if self.args.profile:
            print(
                "det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}".format(
                    dt=np.mean(runtime_profile["dt"]),
                    pt=np.mean(runtime_profile["pt"]),
                    pn=np.mean(runtime_profile["pn"]),
                )
            )

        return pose


class DetectionLoader:

    def __init__(self, detector, cfg, opt):
        self.cfg = cfg
        self.opt = opt
        self.device = opt.device
        self.detector = detector

        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE

        self._sigma = cfg.DATA_PRESET.SIGMA

        if cfg.DATA_PRESET.TYPE == "simple":
            pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
            self.transformation = SimpleTransform(
                pose_dataset,
                scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0,
                sigma=self._sigma,
                train=False,
                add_dpg=False,
                gpu_device=self.device,
            )
        elif cfg.DATA_PRESET.TYPE == "simple_smpl":
            from easydict import EasyDict as edict

            dummpy_set = edict(
                {
                    "joint_pairs_17": None,
                    "joint_pairs_24": None,
                    "joint_pairs_29": None,
                    "bbox_3d_shape": (2.2, 2.2, 2.2),
                }
            )
            self.transformation = SimpleTransform3DSMPL(
                dummpy_set,
                scale_factor=cfg.DATASET.SCALE_FACTOR,
                color_factor=cfg.DATASET.COLOR_FACTOR,
                occlusion=cfg.DATASET.OCCLUSION,
                input_size=cfg.MODEL.IMAGE_SIZE,
                output_size=cfg.MODEL.HEATMAP_SIZE,
                depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
                bbox_3d_shape=(2.2, 2, 2, 2.2),
                rot=cfg.DATASET.ROT_FACTOR,
                sigma=cfg.MODEL.EXTRA.SIGMA,
                train=False,
                add_dpg=False,
                gpu_device=self.device,
                loss_type=cfg.LOSS["TYPE"],
            )

        self.image = (None, None, None, None)
        self.det = (None, None, None, None, None, None, None)
        self.pose = (None, None, None, None, None, None, None)

    def process(self, im_name, image):
        self.image_preprocess(im_name, image)
        self.image_detection()
        self.image_postprocess()
        return self

    def image_preprocess(self, im_name, image):
        img = self.detector.image_preprocess(image)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        orig_img = image
        im_dim = orig_img.shape[1], orig_img.shape[0]

        with torch.no_grad():
            im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

        self.image = (img, orig_img, im_name, im_dim)

    def image_detection(self):
        imgs, orig_imgs, im_names, im_dim_list = self.image
        if imgs is None:
            self.det = (None, None, None, None, None, None, None)
            return

        with torch.no_grad():
            dets = self.detector.images_detection(imgs, im_dim_list)
            if isinstance(dets, int) or dets.shape[0] == 0:
                self.det = (orig_imgs, im_names, None, None, None, None, None)
                return
            if isinstance(dets, np.ndarray):
                dets = torch.from_numpy(dets)
            dets = dets.cpu()
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]
            ids = torch.zeros(scores.shape)

        boxes = boxes[dets[:, 0] == 0]
        if isinstance(boxes, int) or boxes.shape[0] == 0:
            self.det = (orig_imgs, im_names, None, None, None, None, None)
            return
        inps = torch.zeros(boxes.size(0), 3, *self._input_size)
        cropped_boxes = torch.zeros(boxes.size(0), 4)

        self.det = (
            orig_imgs,
            im_names,
            boxes,
            scores[dets[:, 0] == 0],
            ids[dets[:, 0] == 0],
            inps,
            cropped_boxes,
        )

    def image_postprocess(self):
        with torch.no_grad():
            (orig_img, im_name, boxes, scores, ids, inps, cropped_boxes) = self.det
            if orig_img is None:
                self.pose = (None, None, None, None, None, None, None)
                return
            if boxes is None or boxes.nelement() == 0:
                self.pose = (None, orig_img, im_name, boxes, scores, ids, None)
                return

            for i, box in enumerate(boxes):
                inps[i], cropped_box = self.transformation.test_transform(orig_img, box)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)

            self.pose = (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes)

    def read(self):
        return self.pose


class DataWriter:

    def __init__(self, cfg, opt):
        self.cfg = cfg
        self.opt = opt

        self.eval_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        self.item = (None, None, None, None, None, None, None)

        if opt.save_img:
            if not os.path.exists(opt.outputpath):
                os.makedirs(opt.outputpath)

        if opt.pose_flow:
            from trackers.PoseFlow.poseflow_infer import PoseFlowWrapper

            self.pose_flow_wrapper = PoseFlowWrapper(
                save_path=os.path.join(opt.outputpath, "poseflow")
            )

        if self.opt.save_img or self.opt.vis:
            loss_type = self.cfg.DATA_PRESET.get("LOSS_TYPE", "MSELoss")
            num_joints = self.cfg.DATA_PRESET.NUM_JOINTS
            if loss_type == "MSELoss":
                self.vis_thres = [0.05] * num_joints
            elif "JointRegression" in loss_type:
                self.vis_thres = [0.05] * num_joints
            elif loss_type == "Combined":
                if num_joints == 68:
                    hand_face_num = 42
                else:
                    hand_face_num = 110
                self.vis_thres = [0.05] * (num_joints - hand_face_num) + [
                    0.05
                ] * hand_face_num

        self.use_heatmap_loss = (
            self.cfg.DATA_PRESET.get("LOSS_TYPE", "MSELoss") == "MSELoss"
        )

    def start(self):
        return self.update()

    def update(self):
        norm_type = self.cfg.LOSS.get("NORM_TYPE", None)
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE

        (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name) = self.item
        if orig_img is None:
            return {"imgname": im_name, "result": []}

        orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
        self.orig_img = orig_img
        if boxes is None or len(boxes) == 0:
            return {"imgname": im_name, "result": []}
        else:
            assert hm_data.dim() == 4

            face_hand_num = 110
            if hm_data.size()[1] == 136:
                self.eval_joints = [*range(0, 136)]
            elif hm_data.size()[1] == 26:
                self.eval_joints = [*range(0, 26)]
            elif hm_data.size()[1] == 133:
                self.eval_joints = [*range(0, 133)]
            elif hm_data.size()[1] == 68:
                face_hand_num = 42
                self.eval_joints = [*range(0, 68)]
            elif hm_data.size()[1] == 21:
                self.eval_joints = [*range(0, 21)]
            pose_coords = []
            pose_scores = []
            for i in range(hm_data.shape[0]):
                bbox = cropped_boxes[i].tolist()
                if isinstance(self.heatmap_to_coord, list):
                    (
                        pose_coords_body_foot,
                        pose_scores_body_foot,
                    ) = self.heatmap_to_coord[0](
                        hm_data[i][self.eval_joints[:-face_hand_num]],
                        bbox,
                        hm_shape=hm_size,
                        norm_type=norm_type,
                    )
                    (
                        pose_coords_face_hand,
                        pose_scores_face_hand,
                    ) = self.heatmap_to_coord[1](
                        hm_data[i][self.eval_joints[-face_hand_num:]],
                        bbox,
                        hm_shape=hm_size,
                        norm_type=norm_type,
                    )
                    pose_coord = np.concatenate(
                        (pose_coords_body_foot, pose_coords_face_hand), axis=0
                    )
                    pose_score = np.concatenate(
                        (pose_scores_body_foot, pose_scores_face_hand), axis=0
                    )
                else:
                    pose_coord, pose_score = self.heatmap_to_coord(
                        hm_data[i][self.eval_joints],
                        bbox,
                        hm_shape=hm_size,
                        norm_type=norm_type,
                    )
                pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
            preds_img = torch.cat(pose_coords)
            preds_scores = torch.cat(pose_scores)
            if not self.opt.pose_track:
                boxes, scores, ids, preds_img, preds_scores, pick_ids = pose_nms(
                    boxes,
                    scores,
                    ids,
                    preds_img,
                    preds_scores,
                    self.opt.min_box_area,
                    use_heatmap_loss=self.use_heatmap_loss,
                )

            _result = []
            for k in range(len(scores)):
                _result.append(
                    {
                        "keypoints": preds_img[k],
                        "kp_score": preds_scores[k],
                        "proposal_score": torch.mean(preds_scores[k])
                        + scores[k]
                        + 1.25 * max(preds_scores[k]),
                        "idx": ids[k],
                        "box": [
                            boxes[k][0],
                            boxes[k][1],
                            boxes[k][2] - boxes[k][0],
                            boxes[k][3] - boxes[k][1],
                        ],
                    }
                )
            _result = sorted(_result, key=lambda _x: -_x["proposal_score"].item())

            result = {"imgname": im_name, "result": _result}

            if self.opt.pose_flow:
                poseflow_result = self.pose_flow_wrapper.step(orig_img, result)
                for i in range(len(poseflow_result)):
                    result["result"][i]["idx"] = poseflow_result[i]["idx"]

            write_json(
                [result],
                self.opt.outputpath,
                form=self.opt.format,
                for_eval=self.opt.eval,
                outputfile=im_name[:-4] + ".json",
            )
            if self.opt.save_img or self.opt.vis:
                if hm_data.size()[1] == 49:
                    from alphapose.utils.vis import vis_frame_dense as vis_frame
                elif self.opt.vis_fast:
                    from alphapose.utils.vis import vis_frame_fast as vis_frame
                else:
                    from alphapose.utils.vis import vis_frame
                img = vis_frame(orig_img, result, self.opt, self.vis_thres)
                self.write_image(img, im_name, stream=None)

            return result

    def save(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name):
        self.item = (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name)

    def write_image(self, img, im_name, stream=None):
        import cv2

        if self.opt.vis:
            cv2.imshow("AlphaPose Demo", img)
            cv2.waitKey(30)
        if self.opt.save_img:
            cv2.imwrite(os.path.join(self.opt.outputpath, im_name), img)
