"""QGIS plugin lifecycle and Processing provider registration."""

from qgis.core import QgsApplication

from .provider import DstFusionProvider


class DstFusionPlugin:
    """Register the burned-area Processing provider with QGIS."""

    def __init__(self, iface):
        self.iface = iface
        self.provider = None

    def initGui(self):
        """Register the provider when QGIS enables the plugin."""
        if self.provider is not None:
            return

        provider = DstFusionProvider()
        if not QgsApplication.processingRegistry().addProvider(provider):
            raise RuntimeError("Could not register the burned-area Processing provider.")
        self.provider = provider

    def unload(self):
        """Unregister the provider when QGIS disables the plugin."""
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
