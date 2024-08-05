import couplatis


def test_torch_env():
    assert couplatis.torch.__version__.startswith("2.4")
