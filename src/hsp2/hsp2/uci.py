from collections import defaultdict
import warnings

from .model import Model


class UCI(Model):
    def __init__(self) -> None:
        warnings.warn(
            """
*
* UCI class is deprecated, use Model class in hsp2.model instead.
* The UCI class will be deleted sometime in the future.
*
""",
            DeprecationWarning,
        )
        super().__init__()
        self.uci = defaultdict(dict)
