from software_bench.evaluation.parsers import SimplePytestParser


def test_one_suite_log_can_report_multiple_oracle_test_ids() -> None:
    output = """
    tests/test_feature.py::test_new PASSED
    FAILED tests/test_regression.py::test_old
    unrelated summary line
    """

    statuses = SimplePytestParser().parse(output)

    assert statuses == {
        "tests/test_feature.py::test_new": "PASSED",
        "tests/test_regression.py::test_old": "FAILED",
    }
