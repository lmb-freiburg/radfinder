import pandas as pd


def ok(v):
    if pd.isna(v):
        return False
    if isinstance(v, str):
        if v.strip() == "":
            return False
        if v.lower().strip() == "nan":
            return False
    return True
