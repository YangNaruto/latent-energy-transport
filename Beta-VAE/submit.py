# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Submit a function to be run either locally or in a computing cluster."""

import copy
import inspect
import os
import pathlib
import pickle
import platform
import pprint
import re
import shutil
import sys
import time
import traceback
import json
from enum import Enum


class SubmitTarget(Enum):
    """The target where the function should be run.

    LOCAL: Run it locally.
    """
    LOCAL = 1


class PathType(Enum):
    """Determines in which format should a path be formatted.

    WINDOWS: Format with Windows style.
    LINUX: Format with Linux/Posix style.
    AUTO: Use current OS type to select either WINDOWS or LINUX.
    """
    WINDOWS = 1
    LINUX = 2
    AUTO = 3


class PlatformExtras:
    """A mixed bag of values used by dnnlib heuristics.

    Attributes:

        data_reader_buffer_size: Used by DataReader to size internal shared memory buffers.
        data_reader_process_count: Number of worker processes to spawn (zero for single thread operation)
    """
    def __init__(self):
        self.data_reader_buffer_size = 1<<30    # 1 GB
        self.data_reader_process_count = 0      # single threaded default


_user_name_override = None


def get_path_from_template(path_template: str, path_type: PathType = PathType.AUTO) -> str:
    """Replace tags in the given path template and return either Windows or Linux formatted path."""
    # automatically select path type depending on running OS
    if path_type == PathType.AUTO:
        if platform.system() == "Windows":
            path_type = PathType.WINDOWS
        elif platform.system() == "Linux":
            path_type = PathType.LINUX
        else:
            raise RuntimeError("Unknown platform")

    path_template = path_template.replace("<USERNAME>", get_user_name())

    # return correctly formatted path
    if path_type == PathType.WINDOWS:
        return str(pathlib.PureWindowsPath(path_template))
    elif path_type == PathType.LINUX:
        return str(pathlib.PurePosixPath(path_template))
    else:
        raise RuntimeError("Unknown platform")


def get_template_from_path(path: str) -> str:
    """Convert a normal path back to its template representation."""
    path = path.replace("\\", "/")
    return path


def convert_path(path: str, path_type: PathType = PathType.AUTO) -> str:
    """Convert a normal path to template and the convert it back to a normal path with given path type."""
    path_template = get_template_from_path(path)
    path = get_path_from_template(path_template, path_type)
    return path


def set_user_name_override(name: str) -> None:
    """Set the global username override value."""
    global _user_name_override
    _user_name_override = name


def get_user_name():
    """Get the current user name."""
    if _user_name_override is not None:
        return _user_name_override
    elif platform.system() == "Windows":
        return os.getlogin()
    elif platform.system() == "Linux":
        try:
            import pwd
            return pwd.getpwuid(os.geteuid()).pw_name
        except:
            return "unknown"
    else:
        raise RuntimeError("Unknown platform")



def _create_run_dir_local(run_dir_root, run_desc) -> str:
    """Create a new run dir with increasing ID number at the start."""
    run_dir_root = get_path_from_template(run_dir_root, PathType.AUTO)

    if not os.path.exists(run_dir_root):
        os.makedirs(run_dir_root)

    run_id = _get_next_run_id_local(run_dir_root)
    run_name = "{0:05d}-{1}".format(run_id, run_desc)
    run_dir = os.path.join(run_dir_root, run_name)

    if os.path.exists(run_dir):
        raise RuntimeError("The run dir already exists! ({0})".format(run_dir))

    os.makedirs(run_dir)

    return run_dir


def _get_next_run_id_local(run_dir_root: str) -> int:
    """Reads all directory names in a given directory (non-recursive) and returns the next (increasing) run id. Assumes IDs are numbers at the start of the directory names."""
    dir_names = [d for d in os.listdir(run_dir_root) if os.path.isdir(os.path.join(run_dir_root, d))]
    r = re.compile("^\\d+")  # match one or more digits at the start of the string
    run_id = 0

    for dir_name in dir_names:
        m = r.match(dir_name)

        if m is not None:
            i = int(m.group())
            run_id = max(run_id, i + 1)

    return run_id


def copy_files_and_create_dirs(files) -> None:
    """Takes in a list of tuples of (src, dst) paths and copies files.
    Will create all necessary directories."""
    for file in files:
        target_dir_name = os.path.dirname(file[1])

        # will create all intermediate-level directories
        if not os.path.exists(target_dir_name):
            os.makedirs(target_dir_name)

        shutil.copyfile(file[0], file[1])


def _copy_dir(files, run_dir):
    src = os.path.join(run_dir, 'src')
    if not os.path.exists(src):
        os.makedirs(src)
    for file_name in files:
        if os.path.isdir(file_name):
            shutil.copytree(file_name, os.path.join(src, file_name))
        else:
            shutil.copyfile(file_name, os.path.join(src, file_name))

def _save_args(run_dir, file_name, args):
    with open(os.path.join(run_dir, file_name), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

