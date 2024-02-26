# this is a code remnant that lays out a manually created set of objects
# in order to use this appropriate libs must be loaded but this does not yet do so


# now add a simple table 
data_table = np.asarray([ [ 0.0, 5.0, 10.0], [10.0, 15.0, 20.0], [20.0, 25.0, 30.0], [30.0, 35.0, 40.0] ], dtype= "float32")
dm = DataMatrix('dm', river, data_table)
dm.add_op_tokens()
# 2d lookup
dma = DataMatrixLookup('dma', river, dm.state_path, 2, 17.5, 1, 6.8, 1, 0.0)
dma.add_op_tokens()
# 1.5d lookup
#dma = DataMatrixLookup('dma', river, dm.state_path, 3, 17.5, 1, 1, 1, 0.0)
#dma.add_op_tokens()

facility = ModelObject('facility', river)

Qintake = Equation('Qintake', facility, "Qin * 1.0")
Qintake.add_op_tokens()
# a flowby
flowby = Equation('flowby', facility, "Qintake * 0.9")
flowby.add_op_tokens()
# add a withdrawal equation 
# we use "3.0 + 0.0" because the equation parser fails on a single factor (number of variable)
# so we have to tweak that.  However, we need to handle constants separately, and also if we see a 
# single variable equation (such as Qup = Qhydr) we need to rewrite that to a input anyhow for speed
wd_mgd = Equation('wd_mgd', facility, "3.0 + 0.0")
wd_mgd.add_op_tokens() 
# Runit - unit area runoff
Runit = Equation('Runit', facility, "Qin / 592.717")
Runit.add_op_tokens()
# add local subwatersheds to test scalability
"""
for k in range(10):
    subshed_name = 'sw' + str(k)
    upstream_name = 'sw' + str(k-1)
    Qout_eqn = str(25*random.random()) + " * Runit "
    if k > 0:
        Qout_eqn = Qout_eqn + " + " + upstream_name + "_Qout"
    Qout_ss = Equation(subshed_name + "_Qout", facility, eqn)
    Qout_ss.add_op_tokens()
# now add the output of the final tributary to the inflow to this one
Qtotal = Equation("Qtotal", facility, "Qin + " + Qout_ss.name)
Qtotal.tokenize()
"""
# add random ops to test scalability
# add a series of rando equations 
"""
c=["flowby", "wd_mgd", "Qintake"]
for k in range(10000):
    eqn = str(25*random.random()) + " * " + c[round((2*random.random()))]
    newq = Equation('eq' + str(k), facility, eqn)
    newq.add_op_tokens()
"""
# now connect the wd_mgd back to the river with a direct link.  
# This is not how we'll do it for most simulations as there may be multiple inputs but will do for now
hydr = ModelObject('HYDR', river)
hydr.add_op_tokens()
O1 = ModelLinkage('O1', hydr, wd_mgd.state_path, 2)
O1.add_op_tokens()

