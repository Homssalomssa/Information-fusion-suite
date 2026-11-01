# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for QGIS plugin lifecycle and Processing provider registration."""

from unittest.mock import Mock

from possibility_fusion import plugin as plugin_module


def test_plugin_registers_provider_once_and_removes_it_on_unload(monkeypatch):
    registry = Mock()
    registry.addProvider.return_value = True
    provider = object()
    provider_factory = Mock(return_value=provider)
    monkeypatch.setattr(
        plugin_module,
        "QgsApplication",
        Mock(processingRegistry=Mock(return_value=registry)),
    )
    monkeypatch.setattr(plugin_module, "PossibilityFusionProvider", provider_factory)
    plugin = plugin_module.PossibilityFusionPlugin(iface=None)

    plugin.initGui()
    plugin.initProcessing()

    provider_factory.assert_called_once_with()
    registry.addProvider.assert_called_once_with(provider)
    assert plugin.provider is provider

    plugin.unload()

    registry.removeProvider.assert_called_once_with(provider)
    assert plugin.provider is None
