"""FoamBench adapter kept independent from any benchmark runner."""

from foambench.corpus import FoamBenchCorpus, import_dataset
from foambench.models import FoamBenchTask
from foambench.workspace import FoamBenchWorkspace

__all__ = ["FoamBenchCorpus", "FoamBenchTask", "FoamBenchWorkspace", "import_dataset"]
