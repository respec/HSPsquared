"""
Provides tools to support parsing of SPEC-ACTIONS blocks from UCI files.
"""
import pandas as pd

def specactions_parse(info, llines):
    store, parse, path, *_ = info
    lines = iter(llines)
    # Notes:
    # - Only "classic" special actions are currently active.
    # - Other type of SA are recognized by the parser, but not stored in hdf5
    # - The condition shows parent IF-THEN-ELSE entries if applicable
    #   - Each action "head_[action type]" should include an "condition"
    #     column to match with condition expression if applicable
    #   - The condition matches an index in /SPECACTIONS/conditions table
    sa_actions = []  # referred to as "classic" in old HSPF code comments
    head_actions = [
        "OPTYP", "RANGE1", "RANGE2", "DC", "DS", "YR", "MO", "DA",
        "HR", "MN", "D", "T", "VARI", "S1", "S2", "AC", "VALUE",
        "TC", "TS", "NUM", "condition",
    ]
    sa_mult = []
    head_mult = []
    sa_uvquan = []
    head_uvquan = []
    open_conditions = []
    sa_distrb = []
    head_distrb = []
    sa_uvname = []
    head_uvname = []
    sa_conditions = []
    head_conditions = ["cond_id", "parent_id", "sibling_id", "expression"]
    active_condition = -1
    for line in lines:
        if line[2:5] == "MULT":
            sa_mult.append(line)
        elif line[2:8] == "UVQUAN":
            sa_uvquan.append(line)
        elif line[2:8] == "DISTRB":
            sa_distrb.append(line)
        elif line[2:8] == "UVNAME":
            sa_uvname.append(line)
        # - IF statements may span multiple lines, so need to 
        #   continue to parse till a "THEN" is reached
        # - The variable to evaluate in a IF-THEN MUST BE A UVQUAN
        #   since UVQUAN must refer to a variable and UVQUAN is the ONLY
        #   allowable 
        #  - IF may appear anywhere on the line as long as there are only
        #   blanks preceding IF
        # - Any IF/ELSE statement may have a "condition"
        #   like AND OR at the end of an "IF" line, 
        #   and that indicates continuing to the next line
        # - Do the IF-THEN parsing built off the equation parser
        #   using Tim's assembled CONDTIONAL statement 
        #   to populate the spec actions exec op_tokens
        #   Reminder: special action allows 3 kinds of parentheses
        #     do a global search and replace all to ()
        elif (line.strip()[:2] == "IF") or (line.strip()[:7] == "ELSE IF"):
            # now we have at least 1 prior condition (maybe the opening IF)
            line = get_ifs(lines, line, 'THEN') 
            d = parseD(line, parse["SPEC-ACTIONS", "conditions"])
            # IF cant have siblings, only ELSE/ELSE IF can
            if not (line.strip()[:7] == "ELSE IF"):
                sibling_id = -1 
            else:
                sibling_id =  sa_conditions[-1,]['cond_id']
            d['sibling_id'] = sibling_id
            d["parent_id"] = specl_get_parent_condition(open_conditions)
            open_conditions.append(d["cond_id"])
            sa_conditions.append(line)
        elif line.strip()[:4] == "ELSE":
            # now we have at least 1 prior condition (maybe the opening IF)
            d = parseD(line, parse["SPEC-ACTIONS", "conditions"])
            sibling_id =  sa_conditions[-1,]['cond_id']
            d['sibling_id'] = sibling_id
            d["parent_id"] = parent_condition
        elif line.strip() == "END IF":
            #print("found END IF")
            # must replace this with a stack to push/pop
            # in order to track nested conditions.
            if (len(open_conditions) > 0):
                open_conditions.pop()
        else:
            # ACTIONS block
            # todo: TIm has a single function that parses a line
            d = parseD(line, parse["SPEC-ACTIONS", "ACTIONS"])
            d["parent_id"] = specl_get_parent_condition(open_conditions)
            sa_actions.append(d)
    
    if sa_actions:
        dfftable = pd.DataFrame(sa_actions, columns=head_actions).replace("na", "")
        dfftable.to_hdf(store, key=f"/SPEC_ACTIONS/ACTIONS", data_columns=True)
    if sa_conditions:
        dfftable = pd.DataFrame(sa_conditions, columns=head_conditions).replace("na", "")
        # indicate this as a lower case since it is NOT an actual TABLE in HSPF
        dfftable.to_hdf(store, key=f"/SPEC_ACTIONS/conditions", data_columns=True)


def specl_get_parent_condition(open_conditions):
    # if there are open conditiosn on the stack, then, any new 
    # IF or ELSE IF will use this to find it
    if len(open_conditions) > 0:
        parent_condition = open_conditions[-1]
    else:
        parent_condition = -1
    return(parent_condition)

def get_ifs(lines, line, line_end='THEN'):
    end_ln = len(line_end)
    print("Start line:", line)
    print("********************")
    while line[-end_ln:] != line_end:
        nline = next(lines).strip()
        print(line, nline)
        line = line + " " + nline
    return(line)