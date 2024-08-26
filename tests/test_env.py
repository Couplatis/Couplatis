import torch


def test_torch_env():
    assert torch.__version__.startswith("2.4")
