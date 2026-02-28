def run_net(*args, **kwargs):
    from .runner import run_net as _run_net
    return _run_net(*args, **kwargs)


def test_net(*args, **kwargs):
    from .runner import test_net as _test_net
    return _test_net(*args, **kwargs)


__all__ = ["run_net", "test_net"]
