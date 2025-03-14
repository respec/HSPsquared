# sample script to illustrate running HSP2 with inMem version

uciName = "C://dev//HSPsquared//tests//test10//HSP2results//test10.uci"
wdmName = "C://dev//HSPsquared//tests//test10//HSP2results//test10.wdm"

from readers.utilities.readUCIinMem import readUCI
from readers.utilities.readWDMinMem import readWDM

uciDict = readUCI(uciName,'')
wdmDict = readWDM(wdmName,'')
# combine the 2 dicts
uciDict.update(wdmDict)

from utilities.HSP2mainInMem import main
main(uciDict, saveall=True, jupyterlab=False)

x = uciDict
