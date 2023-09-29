import os, sys

# print('in HSP2_Driver')
command_line = ""
# print('arg count ' + str(len(sys.argv)))
if len(sys.argv) >= 2:
    # see if anything on command line
    command_line = sys.argv[1]
    print('command line' + command_line)

# if given UCI, import to h5 file
# if given WDM, import to h5 file
# if given h5, run HSP2

from PyQt5.QtWidgets import QFileDialog, QApplication

application = QApplication(sys.argv)

if command_line == '':
    file_filter = "Run HDF5 (*.h5) Full Output;;" \
                  "Run HDF5 (*.h5) Light Output;;" \
                  "Import UCI to HDF5 (*.uci);;" \
                  "Import WDM to HDF5 (*.wdm)"
    filename, filetype = QFileDialog.getOpenFileName(None, 'HSP2 Open File...', '', file_filter)
else:
    filename = command_line
    filetype = ''

file_ext = filename[-3:]
dir_name = os.path.dirname(filename)
os.chdir(dir_name)

if file_ext.upper() == "UCI":
    h5_name = filename[:-3] + "h5"
    from HSP2tools.readUCI import readUCI
    readUCI(filename, h5_name)
    # readUCI('HSPF.uci', 'test.h5')

if file_ext.upper() == "WDM":
    h5_name = filename[:-3] + "h5"
    from HSP2tools.readWDM import readWDM
    readWDM(filename, h5_name)
    # readWDM('GRICM.wdm', 'test.h5')
    # readWDM('ZUMBROSCEN.WDM', 'test.h5')

if file_ext.upper() == ".H5":
    from HSP2.main import main
    from HSP2IO.hdf import HDF5
    from HSP2IO.io import IOManager

    hdf5_instance = HDF5(filename)
    io_manager = IOManager(hdf5_instance)

    SaveLevel = True
    if 'Light' in filetype:
        SaveLevel = False
    main(io_manager, saveall=SaveLevel, jupyterlab=False)
    # main('test.h5', saveall=True)




