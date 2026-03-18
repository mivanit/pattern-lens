"""Regression test: all pattern_lens annotations must be resolvable by beartype.

The jaxtyping pytest hook (--jaxtyping-packages=pattern_lens,beartype.beartype)
makes beartype eval() all annotation strings at import time. Any name missing
from module globals (e.g. behind TYPE_CHECKING, or a union string like
"Foo|Bar" that beartype can't parse) causes a runtime failure.

This test scans every function in every pattern_lens submodule and calls
typing.get_type_hints() to verify all annotations resolve.
"""

import importlib
import inspect
import pkgutil
import typing

import pattern_lens


def test_all_annotations_resolvable():
	"""All string annotations in pattern_lens must be eval()-able against module globals."""
	for _importer, modname, _ispkg in pkgutil.walk_packages(
		pattern_lens.__path__, prefix="pattern_lens."
	):
		mod = importlib.import_module(modname)

		for attr_name in dir(mod):
			obj = getattr(mod, attr_name)
			if not callable(obj) or inspect.isclass(obj):
				continue
			if getattr(obj, "__module__", None) != modname:
				continue
			typing.get_type_hints(obj)
