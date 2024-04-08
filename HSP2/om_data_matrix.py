"""
The class DataMatrix is used to translate provide table lookup and interpolation function.
  # we need to check these matrix_vals because the OM system allowed 
  # column headers to be put into the first row as placeholders, and 
  # if those string variables did NOT resolve to something in state it would simply 
  # ignore them, which is sloppy.  See: "Handle legacy header values" in https://github.com/HARPgroup/HSPsquared/issues/26
  # self.op_matrix = [] # this is the final opcoded matrix for runtime
"""
from numba import njit
import numpy as np
from HSP2.state import *
from HSP2.om import *
from HSP2.om_model_object import *

class DataMatrix(ModelObject):
    def __init__(self, name, container = False, model_props = {}):
        super(DataMatrix, self).__init__(name, container, model_props)
        if not DataMatrix.check_properties(model_props):
            raise Exception("DataMatrix requires: " + ','.join(DataMatrix.required_properties()) + " ... process terminated.")
        # check this fitrst, because it may determine how we process the inputted matrix
        self.model_props_parsed = model_props # stash these for debugging the loader process
        self.auto_set_vars = self.handle_prop(model_props, 'autosetvars')
        if type(model_props['matrix']) is str:
            # this is a matrix that uses another matrix for its data source
            self.add_input('matrix', model_props['matrix'], 2, False)
            self.matrix_ix = self.get_dict_state(self.inputs_ix['matrix'])
            # we are accessing a remote matrix.
            data_matrix = self.get_dict_state(self.matrix_ix)
        else: 
            self.matrix = np.asarray(self.handle_prop(model_props, 'matrix')) # gets passed in at creation
            if self.auto_set_vars == 1:
                # we need to extract the first row 
                self.auto_var_names = self.matrix[0] # stash it
                self.matrix = np.delete(self.matrix, 0, 0) # now remove it 
            data_matrix = self.matrix 
            self.matrix_ix = self.ix
        self.mx_type = self.handle_prop(model_props, 'mx_type') 
        if self.mx_type == None:
            # check for valuetype instead - this will go away when we enforce mx_type
            print("Guessing mx_type for", self.name, " from valuetype.  This is deprecated. ")
            self.valuetype = self.handle_prop(model_props, 'valuetype') 
            if self.valuetype != None:
                self.mx_type = int(self.valuetype)
                model_props['mx_type'] = self.mx_type
            print("mx_type set to ", self.mx_type)
        if self.mx_type == None:
            #raise Exception("Matrix", self.name, " has neither mx_type nor valuetype.  Object creation halted. ")
            # note: the drupal hydroImpoundment classes ship storage_stage_area as 'matrix' and overwrite all other stuff.  Which is kinda busted, but is apparently needed by the small impoundments in the old OM model.
            # this is not a huge issue, since we will not use those classes in the new model but forces us to 
            # be OK with missing mx_type during testing.
            print("Matrix", self.name, " has neither mx_type nor valuetype.  This is deprecated. ")
            self.mx_type = 0
        if not DataMatrixLookup.check_properties(model_props):
            self.mx_type = 0
        self.optype = 2 # 0 - shell object, 1 - equation, 2 - DataMatrix, 3 - input, 4 - broadcastChannel, 5 - ?
        # tokenized version of the matrix with variable references and constant references
        self.matrix_tokens = []
        # old version on accessor had the below. necessary?
        # self.matrix_tokens = np.zeros(self.nrows * self.ncols, dtype=int ) 
        # set of default values to populate dict_ix
        self.init_matrix_attributes(data_matrix)
        # get and stash the variable name (could be numeric?) inputs for row and col lookup var (keycols1 and 2)
        self.keycol1 = self.handle_prop(model_props, 'keycol1', True)
        self.keycol2 = self.handle_prop(model_props, 'keycol2', True)
        if not self.mx_type > 0:
            # just a matrix of values, return.
            return 
        self.lu_type1 = int(self.handle_prop(model_props, 'lutype1', True ))
        if (self.mx_type > 1):
            self.lu_type2 = int(self.handle_prop(model_props, 'lutype2', True ) )
        else:
            self.key2_ix = 0
            self.lu_type2 = 0
    
    def init_matrix_attributes(self, data_matrix):
        self.nrows = data_matrix.shape[0]
        self.ncols = data_matrix.shape[1]
        self.matrix_values = np.zeros(data_matrix.shape)
    
    @staticmethod
    def required_properties():
        req_props = super(DataMatrix, DataMatrix).required_properties()
        req_props.extend(['matrix'])
        return req_props
    
    def find_paths(self):
        super().find_paths()
        self.key1_ix = self.constant_or_path(self.keycol1, self.keycol1, False )
        if (self.mx_type > 1):
          self.key2_ix = self.constant_or_path(self.keycol1, self.keycol2, False )
        else:
            self.key2_ix = 0
            self.lu_type2 = 0
        self.paths_found = False # override parent setting until we verify everything
        self.matrix_tokens = [] # reset this in case it is called multiple times
        # Now, we filter for the weird cases that legacy OM allowed 
        header_suspects = 0 # this will tally up how many oddities we get
        for k in range(self.ncols):
            # just check if we can find the columns, and count how many we can find.
            # if this is a 2-d lookup we set 
            m_var = self.matrix[0][k]
            if is_float_digit(m_var) == False:
                # this is a string variable, check if it is a header or actual reference
                var_exists = self.find_var_path(self.matrix[0][k])
                if (var_exists == False):
                    if ( (self.mx_type == 2) and (k == 0) ):
                        # this is fine, 2-d arrays don't even use the 0,0 value
                        # so we set it to a constant value of zero
                        self.matrix[0][k] = 0.0 
                    else:
                        header_suspects = header_suspects + 1 
        print("Checked the first row for headers, header_suspects = ", header_suspects)
        if (header_suspects > 0):
            if (header_suspects == self.ncols):
                print("Removing the header row")
                # we have what looks like a header, discard, but warn that this sucks
                self.matrix = np.delete(self.matrix, 0, 0) # now remove it 
                # reset rows, cols, and value state storage matrix 
                self.init_matrix_attributes(self.matrix)
        for i in range(self.nrows):
            for j in range(self.ncols):
                el_name = 'rowcol_' + str(i) + "_" + str(j)
                el_value = self.matrix[i][j]
                # this registers the constants, or values, as inputs
                self.matrix_tokens.append(self.constant_or_path(el_name, el_value, False) )
        self.paths_found = True
    
    def tokenize(self):
        # call parent method to set basic ops common to all 
        super().tokenize()
        # - insure we have a entity_ix pointing to state_ix
        # - check matrix for string vars and get entity_ix for string variables 
        # - add numerical constants to the state_ix and get the resulting entity_ix
        # - format array of all rows and columns state_ix references 
        # - store array in dict_ix keyed with entity_ix
        # - get entity_ix for lookup key(s)
        # - create tokenized array with entity_ix, lookup types, 
        # renders tokens for high speed execution
        # note: first 3 ops are type, ix, and matrix_ix.  
        #       a normal matrix has ix = matrix_ix, but this allows for a matrix object that just accesses another
        #       matrix object as an input.
        self.ops = self.ops + [self.matrix_ix, self.nrows, self.ncols, self.mx_type, self.key1_ix, self.lu_type1, self.key2_ix, self.lu_type2 ] + self.matrix_tokens
    
    def add_op_tokens(self):
        # this puts the tokens into the global simulation queue 
        # can be customized by subclasses to add multiple lines if needed.
        super().add_op_tokens()
        self.dict_ix[self.ix] = DataFrame(self.matrix_values).to_numpy()


class DataMatrixLookup(DataMatrix):
    # this is just here for info purposes when creating a matrix that has its own accessor 
    @staticmethod
    def required_properties():
        req_props = super(DataMatrixLookup, DataMatrixLookup).required_properties()
        req_props.extend(['mx_type', 'keycol1', 'lutype1', 'keycol2', 'lutype2'])
        return req_props


# njit functions for runtime

@njit
def om_table_lookup(data_table, mx_type, ncols, keyval1, lu_type1, keyval2, lu_type2):
    # mx_type = 0: no lookup, matrix, 1: 1d (default to col 2 as value), 2: 2d (both ways), 3: 1.5d (keyval2 = column index) 
    #  - 1: 1d, look up row based on column 0, return value from column 1
    #  - 2: 2d, look up based on row and column 
    #  - 3: 1.5d, look up/interp row based on column 0, return value from column 
    # lu_type: 0 - exact match; 1 - interpolate values; 2 - stair step
    if mx_type == 1:
        valcol = int(1)
        luval = table_lookup(data_table, keyval1, lu_type1, valcol)
        return luval
    if ( (mx_type == 3) or (lu_type2 == 0) ): # 1.5d (a 2-d with exact match column functions just like a 1.5d )
        valcol = int(keyval2)
        luval = table_lookup(data_table, keyval1, lu_type1, valcol)
        return luval
    # must be a 2-d lookup 
    # if lu_type1 is stair step or exact match, we call the whole row 
    if (lu_type1 == 2):
        row_vals = table_row_lookup(data_table, keyval1, lu_type1)
    elif (lu_type1 == 0):
        row_vals = table_row_lookup(data_table, keyval1, lu_type1)
    else:
        # create an interpolated version of the table 
        row_vals = row_interp(data_table, ncols, keyval1, lu_type1)
        # have to use row zero as the keys for row_vals now cause we will interpolate on those
    row_keys = data_table[0]
    # 1: get value for all columns based on the row interp/match type 
    luval = np.interp(keyval2, row_keys, row_vals)
    # show value at tis point
    return luval

@njit 
def row_interp(data_table, ncols, keyval, lu_type):
    row_vals = data_table[0].copy() # initialize to the first row 
    #print("interping for keyval", keyval, "lutype:", lu_type, "ncols", ncols, "in table", data_table)
    for i in range(ncols):
        row_vals[i] = table_lookup(data_table, keyval, lu_type, i)
    return row_vals

@njit
def table_row_lookup(data_table, keyval, lu_type):
    #print("looking for keyval", keyval, "lutype:", lu_type, "in table", data_table)
    if (lu_type == 2):
        # stair step retrieve whole row 
        idx = (data_table[:, 0][0:][(data_table[:, 0][0:]- keyval) <= 0]).argmax()
    elif (lu_type == 0):
        idx = int(keyval)
    #print("looking for row", idx, "in table", data_table)
    row_vals = data_table[:][0:][idx]
    return row_vals

@njit
def table_lookup(data_table, keyval, lu_type, valcol):
    if lu_type == 2: #stair-step
        idx = (data_table[:, 0][0:][(data_table[:, 0][0:]- keyval) <= 0]).argmax()
        luval = data_table[:, valcol][0:][idx]
    elif lu_type == 1: # interpolate
        luval = np.interp(keyval,data_table[:, 0][0:], data_table[:, valcol][0:])
    elif lu_type == 0: # exact match
        lurow = table_row_lookup(data_table, keyval, lu_type)
        luval = lurow[valcol]
    
    # show value at this point
    return luval


"""
exec_tbl_values() updates the values in the dict_ix state for a data matrix as it may contain named variables 
- todo: determine if time savings can be achieved by storing state references for ONLY variable names in matrices
        Since we currently tokenize every item in the dict_ix and copy it from state_ix every timestep
        regardless of whether it is an actual named variable reference or a static numeric value.
        See if speed increases by storing a series of row,col,ix ops for ONLY those entries that are 
        actual variable references, instead of the current method of storing a ix pointer for every single 
        matrix cell.
"""
@njit(cache=True)
def exec_tbl_values(op, state_ix, dict_ix):
    # this f
    ix = op[1]
    data_matrix = dict_ix[ix]
    nrows = op[3]
    ncols = op[4]
    k = 0
    for i in range(nrows):
        for j in range(ncols):
            # the indices pointing to data columns begin at item 10 in the array
            data_matrix[i][j] = state_ix[op[10 + k]]
            k = k + 1
    dict_ix[ix] = data_matrix 
    return 0.0

@njit(cache=True)
def exec_tbl_eval(op_tokens, op, state_ix, dict_ix):
    # Note: these indices must be adjusted to reflect the number of common op tokens
    # check this first, if it is type = 0, then it is just a matrix, and only needs to be loaded, not evaluated
    #print("Called exec_tbl_eval, with ops=", op)
    ix = op[1]
    dix = op[2]
    tbl_op = op_tokens[dix]
    #print("Target Table ops:", tbl_op)
    nrows = tbl_op[3]
    ncols = tbl_op[4]
    # load lookup infor for this accessor 
    mx_type = op[5] # not used yet, what type of table?  in past this was always 1-d or 2-d 
    key1_ix = op[6]
    #print("ix, dict_ix, mx_type, key1_ix", ix, dix, mx_type, key1_ix)
    lu_type1 = op[7]
    key2_ix = op[8]
    lu_type2 = op[9]
    data_table = dict_ix[dix]
    keyval1 = state_ix[key1_ix]
    if key2_ix != 0:
        keyval2 = state_ix[key2_ix]
    else:
        keyval2 = 0
    #print("keyval1, lu_type1, keyval2, lu_type2, ncols", keyval1, lu_type1, keyval2, lu_type2, ncols)
    result = om_table_lookup(data_table, mx_type, ncols, keyval1, lu_type1, keyval2, lu_type2)
    return result

def debug_tbl_eval(op_tokens, op, state_ix, dict_ix):
    # Note: these indices must be adjusted to reflect the number of common op tokens
    # check this first, if it is type = 0, then it is just a matrix, and only needs to be loaded, not evaluated
    ix = op[1]
    dix = op[2]
    tbl_op = op_tokens[dix]
    print("Target Table ops:", tbl_op)
    nrows = tbl_op[3]
    ncols = tbl_op[4]
    # load lookup infor for this accessor 
    mx_type = op[5] # not used yet, what type of table?  in past this was always 1-d or 2-d 
    key1_ix = op[6]
    print("ix, dict_ix, mx_type, key1_ix", ix, dix, mx_type, key1_ix)
    lu_type1 = op[7]
    key2_ix = op[8]
    lu_type2 = op[9]
    print("lu_type1, key2_ix, lu_type2", lu_type1, key2_ix, lu_type2)
    data_table = dict_ix[dix]
    print("data_table", data_table)
    print("key1_ix, key2_ix", key1_ix, key2_ix)
    keyval1 = state_ix[key1_ix]
    if key2_ix != 0:
        keyval2 = state_ix[key2_ix]
    else:
        keyval2 = 0
    print("key1_ix, key2_ix, keyval1, keyval2", key1_ix, key2_ix, keyval1, keyval2)
    print("keyval1, lu_type1, keyval2, lu_type2, ncols", keyval1, lu_type1, keyval2, lu_type2, ncols)
    result = om_table_lookup(data_table, mx_type, ncols, keyval1, lu_type1, keyval2, lu_type2)
    return result
