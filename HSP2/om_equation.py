"""
The class Equation is used to translate an equation in text string form into a tokenized model op code
The equation will look for variable names inside the equation string (i.e. not numeric, not math operator)
and will then search the local object inputs and the containing object inputs (if object has parent) for 
the variable name in question.  Ultimately, everyting becomes either an operator or a reference to a variable
in the state_ix Dict for runtime execution.
"""
from HSP2.om import *
from HSP2.state import *
from HSP2.om_model_object import *
from numba import njit
class Equation(ModelObject):
    # the following are supplied by the parent class: name, log_path, attribute_path, state_path, inputs
    
    def __init__(self, name, container = False, model_props = {}):
        super(Equation, self).__init__(name, container, model_props)
        self.equation = self.handle_prop(model_props, 'equation') 
        self.ps = False 
        self.ps_names = [] # Intermediate with constants turned into variable references in state_paths
        self.var_ops = [] # keep these separate since the equation functions should not have to handle overhead
        self.optype = 1 # 0 - shell object, 1 - equation, 2 - datamatrix, 3 - input, 4 - broadcastChannel, 5 - ?
        self.non_neg = self.handle_prop(model_props, 'nonnegative', False, 0)
        min_value = self.handle_prop(model_props, 'minvalue', False, 0.0)
        self.min_value_ix = ModelConstant('min_value', self, min_value).ix
        self.deconstruct_eqn()
    
    def handle_prop(self, model_props, prop_name, strict = False, default_value = None ):
        prop_val = super().handle_prop(model_props, prop_name, strict, default_value )
        if (prop_name == 'equation'):
            if type(prop_val) is str:
                return prop_val
            elif prop_val == None:
                # try for equation stored as normal propcode
                prop_val = str(self.handle_prop(model_props, 'value', True))
        return prop_val
    
    def deconstruct_eqn(self):
        exprStack = []
        exprStack[:] = []
        self.ps = deconstruct_equation(self.equation)
        # if ps is empty, we may have gotten a constant, so we will check it, 
        # and create a set of ps [constant] + 0.0 and return
        # if this does not yield ps > 0 we will throw an error
        if (len(self.ps) == 0):
            tps = deconstruct_equation(self.equation + " + 0.0")
            if len(tps) == 1:
               # seemed to have succeeded, try to use this now
               self.ps = tps
               self.equation = self.equation + " + 0.0"
            else:
                raise Exception("Error: Cannot parse equation: " + self.equation + " on object " + self.name + " Halting.") 
        if (len(self.ps) == 1):
            # this is a single set of operands, but we need to check for a solo negative number
            # which will also get returned as one op set, with the first slot a number, then 0, 0
            # the first token *should* be an operator, followed by 2 operands
            if is_float_digit(self.ps[0][0]):
                # if it is longer than 1 character
                if (self.ps[0][1] == 0) and (self.ps[0][2] == 0):
                    tps = deconstruct_equation(" 0.0 " + self.equation)
                    if len(tps) == 1:
                        # seemed to have succeeded, try to use this now
                        self.ps = tps
                        self.equation = " 0.0 " + self.equation
                    else:
                        raise Exception("Error: Cannot parse equation: " + self.equation + " on object " + self.name + " Halting.") 

            #print(exprStack)
    
    def find_paths(self):
        super().find_paths()
        self.paths_found = False # override parent setting until we verify everything
        #return 
        # we now want to check to see that all variables can be found (i.e. path exists) 
        # and we want to create variables for any constants that we have here 
        # do not handle mathematical operators
        self.ps_names = []
        for i in range(len(self.ps)):
            name_op = ["", "", ""]
            name_op[0] = self.ps[i][0]
            for j in range(1,3):
                # range 1,3 counts thru 1 and 2 (not 3, cause python is so array centric it knows you know)
                op_value = self.ps[i][j]
                if op_value == None:
                    # don't need to check these as they are just referring to the stack.
                    continue
                if is_float_digit(op_value):
                    op_name = "_op_" + str(i) + "_" + str(j)
                else:
                    op_name = op_value 
                #print("Checking op set", i, "op", j, ":", op_name)
                # constant_or_path() looks at name and path, since we only have a var name, we must assume 
                # the path is either a sibling or child variable or explicitly added other input, so this should
                # resolve correctly, but we have not tried it
                var_ix = self.constant_or_path(op_name, op_value, False)
                # we now return, trusting that the state_path for each operand 
                # is stored in self.inputs, with the varix saved in self.inputs_ix
                name_op[j] = op_name
            self.ps_names.append(name_op)
        self.paths_found = True
        return
    
    def tokenize_ops(self):
        self.deconstruct_eqn()
        self.var_ops = tokenize_ops(self.ps)
    
    def tokenize_vars(self):
      # now stash the string vars as new state vars
      for j in range(2,len(self.var_ops)):
          if isinstance(self.var_ops[j], int):
              continue # already has been tokenized, so skip ahead
          elif is_float_digit(self.var_ops[j]):
              # must add this to the state array as a constant
              constant_path = self.state_path + '/_ops/_op' + str(j) 
              s_ix = set_state(self.state_ix, self.state_paths, constant_path, float(self.var_ops[j]) )
              self.var_ops[j] = s_ix
          else:
              # this is a variable, must find it's data path index
              var_path = self.find_var_path(self.var_ops[j])
              s_ix = get_state_ix(self.state_ix, self.state_paths, var_path)
              if s_ix == False:
                  print("Error: unknown variable ", self.var_ops[j], "path", var_path, "index", s_ix)
                  print("searched: ", self.state_paths, self.state_ix)
                  return
              else:
                  self.var_ops[j] = s_ix
    
    def tokenize(self):
        # call parent to render standard tokens
        super().tokenize()
        # replaces operators with integer code,
        # and turns the array of 3 value opcode arrays into a single sequential array 
        self.tokenize_ops() 
        # finds the ix value for each operand 
        self.tokenize_vars()
        # renders tokens for high speed execution
        self.ops = self.ops + [self.non_neg, self.min_value_ix] + self.var_ops
 

from pyparsing import (
    Literal,
    Word,
    Group,
    Forward,
    alphas,
    alphanums,
    Regex,
    ParseException,
    CaselessKeyword,
    Suppress,
    delimitedList,
)
import math
import operator

exprStack = []


def push_first(toks):
    exprStack.append(toks[0])


def push_unary_minus(toks):
    for t in toks:
        if t == "-":
            exprStack.append("unary -")
        else:
            break

def deconstruct_equation(eqn):
    """
    We should get really good at using docstrings...

    we parse the equation during readuci/pre-processing and break it into njit'able pieces
    this forms the basis of our object parser code to run at import_uci step 
    """
    results = BNF().parseString(eqn, parseAll=True)
    ps = []
    ep = exprStack
    pre_evaluate_stack(ep[:], ps)
    return ps

def get_operator_token(op):
    # find the numerical token for an operator
    # returns integer value, or 0 if this is not a recorgnized mathematical operator
    if op == '-': opn = 1
    elif op == '+': opn = 2
    elif op == '*': opn = 3
    elif op == '/': opn = 4
    elif op == '^': opn = 5
    else: opn = False
    return opn

def tokenize_ops(ps):
    '''Translates a set of string operands into integer keyed tokens for faster execution.''' 
    tops = [len(ps)] # first token is number of ops
    for i in range(len(ps)):
        op = get_operator_token(ps[i][0])
        # a negative op code indicates null
        # this should cause no confusion since all op codes are references and none are actual values
        if ps[i][1] == None: o1 = -1 
        else: o1 = ps[i][1]
        if ps[i][2] == None: o2 = -1 
        else: o2 = ps[i][2]
        tops.append(op)
        tops.append(o1)
        tops.append(o2)
    return tops

bnf = None


def BNF():
    """
    expop   :: '^'
    multop  :: '*' | '/'
    addop   :: '+' | '-'
    integer :: ['+' | '-'] '0'..'9'+
    atom    :: PI | E | real | fn '(' expr ')' | '(' expr ')'
    factor  :: atom [ expop factor ]*
    term    :: factor [ multop factor ]*
    expr    :: term [ addop term ]*
    """
    global bnf
    if not bnf:
        # use CaselessKeyword for e and pi, to avoid accidentally matching
        # functions that start with 'e' or 'pi' (such as 'exp'); Keyword
        # and CaselessKeyword only match whole words
        e = CaselessKeyword("E")
        pi = CaselessKeyword("PI")
        # fnumber = Combine(Word("+-"+nums, nums) +
        #                    Optional("." + Optional(Word(nums))) +
        #                    Optional(e + Word("+-"+nums, nums)))
        # or use provided pyparsing_common.number, but convert back to str:
        # fnumber = ppc.number().addParseAction(lambda t: str(t[0]))
        fnumber = Regex(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")
        ident = Word(alphas, alphanums + "_$")
        
        plus, minus, mult, div = map(Literal, "+-*/")
        lpar, rpar = map(Suppress, "()")
        addop = plus | minus
        multop = mult | div
        expop = Literal("^")
        
        expr = Forward()
        expr_list = delimitedList(Group(expr))
        # add parse action that replaces the function identifier with a (name, number of args) tuple
        def insert_fn_argcount_tuple(t):
            fn = t.pop(0)
            num_args = len(t[0])
            t.insert(0, (fn, num_args))
        
        fn_call = (ident + lpar - Group(expr_list) + rpar).setParseAction(
            insert_fn_argcount_tuple
        )
        atom = (
            addop[...]
            + (
                (fn_call | pi | e | fnumber | ident).setParseAction(push_first)
                | Group(lpar + expr + rpar)
            )
        ).setParseAction(push_unary_minus)
        
        # by defining exponentiation as "atom [ ^ factor ]..." instead of "atom [ ^ atom ]...", we get right-to-left
        # exponents, instead of left-to-right that is, 2^3^2 = 2^(3^2), not (2^3)^2.
        factor = Forward()
        factor <<= atom + (expop + factor).setParseAction(push_first)[...]
        term = factor + (multop + factor).setParseAction(push_first)[...]
        expr <<= term + (addop + term).setParseAction(push_first)[...]
        bnf = expr
    return bnf


# map operator symbols to corresponding arithmetic operations
epsilon = 1e-12
opn = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "^": operator.pow,
}

fn = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "abs": abs,
    "trunc": int,
    "round": round,
    "sgn": lambda a: -1 if a < -epsilon else 1 if a > epsilon else 0,
    # functionsl with multiple arguments
    "multiply": lambda a, b: a * b,
    "hypot": math.hypot,
    # functions with a variable number of arguments
    "all": lambda *a: all(a),
}

fns = {
    "sin": "math.sin",
    "cos": "math.cos",
    "tan": "math.tan",
    "exp": "math.exp",
    "abs": "abs",
    "trunc": "int",
    "round": "round",
}


def evaluate_stack(s):
    op, num_args = s.pop(), 0
    if isinstance(op, tuple):
        op, num_args = op
    if op == "unary -":
        return -evaluate_stack(s)
    if op in "+-*/^":
        # note: operands are pushed onto the stack in reverse order
        op2 = evaluate_stack(s)
        op1 = evaluate_stack(s)
        return opn[op](op1, op2)
    elif op == "PI":
        return math.pi  # 3.1415926535
    elif op == "E":
        return math.e  # 2.718281828
    elif op in fn:
        # note: args are pushed onto the stack in reverse order
        args = reversed([evaluate_stack(s) for _ in range(num_args)])
        return fn[op](*args)
    elif op[0].isalpha():
        raise Exception("invalid identifier '%s'" % op)
    else:
        # try to evaluate as int first, then as float if int fails
        try:
            return int(op)
        except ValueError:
            return float(op)

def pre_evaluate_stack(s, ps):
    op, num_args = s.pop(), 0
    if isinstance(op, tuple):
        op, num_args = op
    if op == "unary -":
        ps.append([-evaluate_stack(s), 0, 0])
        return 
    if op in "+-*/^":
        # note: operands are pushed onto the stack in reverse order
        op2 = pre_evaluate_stack(s, ps)
        op1 = pre_evaluate_stack(s, ps)
        ps.append([ op, op1, op2])
        return 
    elif op == "PI":
        ps.append([math.pi, 0, 0])  # 3.1415926535
        return 
    elif op == "E":
        ps.append([math.e, 0, 0])  # 2.718281828
        return 
    elif op in fns:
        # note: args are pushed onto the stack in reverse order
        #print("s:", s, "op", op)
        args = []
        for x in range(num_args):
            args.append(pre_evaluate_stack(s, ps))
        args.reverse()
        args.insert(fns[op], 0)
        ps.append(args)
        return 
    elif op[0].isalpha():
        return op
    else:
        # return the operand now
        return op


@njit(cache=True)
def evaluate_eq_ops(op, val1, val2):
    if op == 1:
        #print(val1, " - ", val2)
        result = val1 - val2
        return result
    if op == 2:
        #print(val1, " + ", val2)
        result = val1 + val2
        return result
    if op == 3:
        #print(val1, " * ", val2)
        result = val1 * val2 
        return result
    if op == 4:
        #print(val1, " / ", val2)
        result = val1 / val2 
        return result
    if op == 5:
        #print(val1, " ^ ", val2)
        result = pow(val1, val2) 
        return result
    return 0


@njit
def step_equation(op_token, state_ix):
    op_class = op_token[0] # we actually use this in the calling function, which will decide what 
                      # next level function to use 
    result = 0
    s = np.array([0.0])
    s_ix = -1 # pointer to the top of the stack
    s_len = 1
    # handle special equation settings like "non-negative", etc.
    non_neg = op_token[2]
    min_ix = op_token[3]
    num_ops = op_token[4] # this index is equal to the number of ops common to all classes + 1.  See om_model_object for base ops and adjust
    op_loc = 5 # where do the operators and operands start in op_token
    #print(num_ops, " operations")
    # is the below faster since it avoids a brief loop and a couple ifs for 2 op equations?
    if num_ops == 1:
        result = evaluate_eq_ops(op_token[op_loc], state_ix[op_token[op_loc + 1]], state_ix[op_token[op_loc + 2]])
    else:
        for i in range(num_ops): 
            # the number of ops common to all classes + 1 (the counter for math operators) is offset for this
            # currently 3  (2 common ops (0,1), plus 1 to indicate number of equation operand sets(2), so this is ix 3)      
            op = op_token[op_loc + 3*i]
            t1 = op_token[op_loc + 3*i + 1]
            t2 = op_token[op_loc + 3*i + 2]
            # if val1 or val2 are < 0 this means they are to come from the stack
            # if token is negative, means we need to use a stack value
            #print("s", s)
            if t1 < 0: 
                val1 = s[s_ix]
                s_ix -= 1
            else:
                val1 = state_ix[t1]
            if t2 < 0: 
                val2 = s[s_ix]
                s_ix -= 1
            else:
                val2 = state_ix[t2]
            #print(s_ix, op, val1, val2)
            result = evaluate_eq_ops(op, val1, val2)
            s_ix += 1
            if s_ix >= s_len: 
                s = np.append(s, 0)
                s_len += 1
            s[s_ix] = result
        result = s[s_ix]
    if (non_neg == 1) and (result < 0):
        result = state_ix[min_ix]
    state_ix[op_token[1]] = result
    return True