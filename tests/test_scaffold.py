"""Smoke tests for the pure-Python core package scaffold."""

import importlib


def test_core_modules_import_without_qgis():
    modules = (
        "dst_fusion.core.indices",
        "dst_fusion.core.memberships",
        "dst_fusion.core.possibility",
        "dst_fusion.core.temporal",
    )

    for module_name in modules:
        module = importlib.import_module(module_name)
        assert module.__doc__
