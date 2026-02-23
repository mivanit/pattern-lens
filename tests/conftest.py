"""shared test fixtures"""

import shutil
from pathlib import Path

import pytest

TEMP_DIR = Path("tests/.temp")


@pytest.fixture(autouse=True, scope="session")
def _clean_temp_dir():
	"""Remove stale test artifacts before the test session.

	Tests write to tests/.temp/<test_name>/ and don't clean up after themselves
	(each test uses a unique subdirectory so they don't collide). Wiping the
	directory at session start prevents stale files from a previous run from
	masking regressions -- e.g. a test that checks file existence could pass
	from leftover data even if the code that creates the files is broken.
	"""
	if TEMP_DIR.exists():
		shutil.rmtree(TEMP_DIR)
	TEMP_DIR.mkdir(parents=True, exist_ok=True)
