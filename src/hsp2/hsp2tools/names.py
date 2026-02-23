"""
Functions that should create paths, combined names, etc. should go here
"""

def hsp2_sequence_id(activity, ix):
    s = f"{activity[0]}{int(ix):03d}"
    return(s)