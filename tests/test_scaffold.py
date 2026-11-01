"""Smoke tests for the pure-Python core package scaffold."""

import importlib


def test_core_modules_import_without_qgis():
    modules = (
        "possibility_fusion.core.indices",
        "possibility_fusion.core.memberships",
        "possibility_fusion.core.possibility",
        "possibility_fusion.core.temporal",
    )

    for module_name in modules:
        module = importlib.import_module(module_name)
        assert module.__doc__
