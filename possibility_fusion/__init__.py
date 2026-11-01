"""QGIS plugin entry point for burned-area possibility fusion."""


def classFactory(iface):
    """Create the plugin instance required by QGIS."""
    from .plugin import PossibilityFusionPlugin

    return PossibilityFusionPlugin(iface)
