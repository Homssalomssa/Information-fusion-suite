"""Processing provider registration; algorithms are added in a later milestone."""

from qgis.core import QgsProcessingProvider


class DstFusionProvider(QgsProcessingProvider):
    """Provider shell for the burned-area fusion algorithms."""

    def id(self):
        return "dst_fusion"

    def name(self):
        return self.tr("Burned-area possibility fusion")

    def longName(self):
        return self.name()

    def loadAlgorithms(self):
        """Algorithms will be registered once their core implementations exist."""
