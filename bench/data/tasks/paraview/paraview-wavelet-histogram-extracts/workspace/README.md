# Wavelet histogram extracts

This TaskBundle is derived from ParaView's public `WaveletHistogramExtracts` regression workflow. The original workflow creates a temporal analytic source, a slice, a time annotation, a histogram, a multiview layout, and differently scheduled extractors.

Implement the requested pipeline with ParaView or pvbatch. The original GUI event trace is intentionally not included: the task evaluates a portable output contract rather than replaying a recorded interaction.
