def test_torch_available():
    import torch

    assert torch.cuda.is_available()
