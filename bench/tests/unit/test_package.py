import software_bench


def test_package_exposes_version() -> None:
    assert software_bench.__version__ == "0.0.1"

