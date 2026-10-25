"""House Price Intelligence System.

One end-to-end project covering all five units of the 'Data Science
using Python' (CAPCTD3) syllabus on Bengaluru house price data:
loading and cleaning (I), preprocessing (II), EDA and inference (III),
visualization (IV) and linear regression (V).
"""

__version__ = "1.0.0"

from .pipeline import run_pipeline

__all__ = ["__version__", "run_pipeline"]
