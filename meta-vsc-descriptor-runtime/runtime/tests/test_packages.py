# adapted from pangeo https://github.com/pangeo-data/pangeo-docker-images/blob/master/tests/test_pangeo-notebook.py
import importlib
import subprocess
import warnings

import pytest


packages = [
    # these are problem libraries that don't always seem to import, mostly due
    # to dependencies outside the python world
    "fastai",
    "keras",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",  # scikit-learn
    "tensorflow",
    "torch",  # pytorch
    "faiss",
    "augly",
    "PIL",
    "PIL.Image",
]


@pytest.mark.parametrize("package_name", packages, ids=packages)
def test_import(package_name):
    importlib.import_module(package_name)


def test_gpu_packages():
    try:
        subprocess.check_call(["nvidia-smi"])

        import torch

        assert torch.cuda.is_available()

        import tensorflow as tf

        assert tf.test.is_built_with_cuda()
        assert tf.config.list_physical_devices("GPU")

        import faiss
        assert faiss.get_num_gpus() > 0

    except FileNotFoundError:
        warnings.warn(
            "Skipping GPU import tests since nvidia-smi is not present on test machine."
        )
