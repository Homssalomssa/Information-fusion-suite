# SPDX-License-Identifier: GPL-2.0-or-later

"""Processing provider registration; algorithms are added in a later milestone."""

from qgis.core import QgsProcessingProvider

from .algorithms.fuse_burned_area import FuseBurnedAreaAlgorithm


class PossibilityFusionProvider(QgsProcessingProvider):
    """Provider shell for the burned-area fusion algorithms."""

    def id(self):
        return "possibility_fusion"

    def name(self):
        return self.tr("Burned-area possibility fusion")

    def longName(self):
        return self.name()

    def loadAlgorithms(self):
        """Register burned-area Processing algorithms."""
        self.addAlgorithm(FuseBurnedAreaAlgorithm())
