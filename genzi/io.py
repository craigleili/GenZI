from pathlib import Path
import json
import os
import os.path as osp
import re
import shutil
import numpy as np
import yaml


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def may_create_folder(folder_path):
    if not osp.exists(folder_path):
        oldmask = os.umask(000)
        os.makedirs(folder_path, mode=0o777)
        os.umask(oldmask)
        return True
    return False


def make_clean_folder(folder_path):
    success = may_create_folder(folder_path)
    if not success:
        shutil.rmtree(folder_path)
        may_create_folder(folder_path)


def parent_folder(file_path):
    return str(Path(file_path).parent)


def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [
        convert(c) for c in re.split("([0-9]+)", key) if len(c) > 0
    ]
    return sorted(file_list_ordered, key=alphanum_key)


def list_files(folder_path, name_filter, alphanum_sort=False, full_path=False):
    file_list = [
        p.name for p in list(Path(folder_path).glob(name_filter)) if p.is_file()
    ]
    if alphanum_sort:
        file_list = sorted_alphanum(file_list)
    else:
        file_list = sorted(file_list)
    if full_path:
        file_list = [osp.join(folder_path, fn) for fn in file_list]
    return file_list


def list_folders(folder_path, name_filter="*", alphanum_sort=False, full_path=False):
    folder_list = [
        p.name for p in list(Path(folder_path).glob(name_filter)) if p.is_dir()
    ]
    if alphanum_sort:
        folder_list = sorted_alphanum(folder_list)
    else:
        folder_list = sorted(folder_list)
    if full_path:
        folder_list = [osp.join(folder_path, fn) for fn in folder_list]
    return folder_list


def read_lines(file_path):
    with open(file_path, "r") as fin:
        lines = [line.strip() for line in fin.readlines() if len(line.strip()) > 0]
    return lines


def read_strings(file_path):
    with open(file_path, "r") as fin:
        ret = fin.readlines()
    return "".join(ret)


def read_json(filepath):
    with open(filepath, "r") as fh:
        ret = json.load(fh)
    return ret


def write_json(filepath, data):
    assert isinstance(data, (dict, tuple, list))
    with open(filepath, "w") as fh:
        fh.write(json.dumps(data))


def read_yaml(filepath):
    with open(filepath, "r") as fh:
        ret = yaml.safe_load(fh)
    return ret


def write_yaml(filepath, data, flow_style=False):
    assert isinstance(data, (dict, tuple, list))
    with open(filepath, "w") as fh:
        yaml.dump(data, fh, default_flow_style=flow_style)


def read_image(filepath, to_chw=False):
    import cv2

    img = cv2.imread(filepath)[..., ::-1]
    if img.shape[-1] == 4:
        img = img[..., :-1]
    assert img.shape[-1] == 3
    if to_chw:
        return np.transpose(img, (2, 0, 1))
    else:
        return img


def write_image(filepath, image, scale=255, shape=None):
    import cv2

    assert image.ndim == 3
    if image.shape[-1] != 3 and image.shape[-1] != 4:
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        image = np.asarray(image * scale, dtype=np.uint8)
    if shape is not None:
        image = cv2.resize(image, shape, interpolation=cv2.INTER_AREA)
    may_create_folder(parent_folder(filepath))
    cv2.imwrite(filepath, image[..., ::-1])
    return filepath
