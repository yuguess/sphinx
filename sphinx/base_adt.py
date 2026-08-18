from typing import List 
import pandas as pd

AlphaS = str

# universe id string, like T30R20, T55R1
UnivS = str

# string format symbol
SymS = str
SymS_L = List[SymS]

# date string format like yyyy-mm-dd
DateS = str
DateS_L = List[DateS]

Strs = List[str]


# column is univ, index is time index in FREQ
PanelDF = pd.DataFrame