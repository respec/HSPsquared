# sample script to illustrate running HSP2 with inMem version

# reads UCI and WDMs into dictionary of pandas dataframes,
# executes HSP2 off the in-memory dictionary of pandas dataframes,
# writes output to same dictionary of pandas dataframes

uciName = "C://dev//HSPsquared//tests//test10//HSP2results//test10.uci"

from hsp2.hsp2tools.readUCIinMem import readUCI
from hsp2.hsp2tools.readWDMinMem import readWDM
import os

uciDict = readUCI(uciName,'')
uci_path = os.path.dirname(uciName)

input_file_missing = False
filesDf = uciDict['/CONTROL/FILES']
for index, row in filesDf.iterrows():
    if row['TYPE'][0:3] == 'WDM':
        wdmName = row['NAME']
        wdmName = os.path.join(uci_path, wdmName)

        if os.path.isfile(wdmName):
            wdmDict = readWDM(wdmName,'')
            # combine the 2 dicts
            uciDict.update(wdmDict)
        else:
            input_file_missing = True

if input_file_missing:
    print("Cant run " + uciName + ", wdm file missing")
else:
    from hsp2.hsp2.HSP2mainInMem import main
    main(uciDict, saveall=True, jupyterlab=False)

x = uciDict
