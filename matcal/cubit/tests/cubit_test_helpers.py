import unittest

from matcal.cubit.cubit_runner import get_cubit_path


def skip_if_cubit_path_not_registered():
    try:
        get_cubit_path()
    except (KeyError, RuntimeError) as exc:
        raise unittest.SkipTest(
            f"Cubit executable path is not registered. Skipping Cubit execution test. {exc}"
        ) from exc
