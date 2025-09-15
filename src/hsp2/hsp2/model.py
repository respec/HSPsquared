from collections import defaultdict

from pandas import DataFrame


class Model:
    def __init__(self) -> None:
        self.model = defaultdict(dict)
        self.ddlinks = defaultdict(list)
        self.ddmasslinks = defaultdict(list)
        self.ddext_sources = defaultdict(list)
        self.ddgener = defaultdict(dict)
        self.siminfo = {}
        self.opseq = DataFrame()
        self.ftables = {}
        self.specactions = {}
        self.monthdata = None
