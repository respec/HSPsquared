import os
import sys

# print('in HSP2_Driver')
command_line = ""
# print('arg count ' + str(len(sys.argv)))
if len(sys.argv) >= 2:
    # see if anything on command line
    command_line = sys.argv[1]
    print("command line" + command_line)

# if given UCI, import to h5 file
# if given WDM, import to h5 file
# if given h5, run HSP2

from PyQt5.QtWidgets import QApplication, QFileDialog

application = QApplication(sys.argv)

if command_line == "":
    file_filter = (
        "Run HDF5 Full Output (*.h5);;"
        "Run HDF5 Light Output (*.h5);;"
        "Import UCI to HDF5 (*.uci);;"
        "Import WDM to HDF5 (*.wdm)"
    )
    filename, filetype = QFileDialog.getOpenFileName(
        None, "HSP2 Open File...", "", file_filter
    )
else:
    filename = command_line
    filetype = ""

file_ext = os.path.splitext(filename)[1].lstrip(".")
dir_name = os.path.dirname(filename)
os.chdir(dir_name)

print(file_ext)
if file_ext.upper() == "UCI":
    h5_name = filename[:-3] + "h5"
    from hsp2.hsp2io.convert_control_files import convert_control_files

    convert_control_parameters(filename, h5_name)

if file_ext.upper() == "WDM":
    h5_name = filename[:-3] + "h5"
    from hsp2.hsp2io.convert_time_series import convert_time_series

    convert_time_series(filename, h5_name)
    # readWDM('GRICM.wdm', 'test.h5')
    # readWDM('ZUMBROSCEN.WDM', 'test.h5')

if file_ext.upper() == ".H5":
    from hsp2.hsp2io.io import IOManager
    from hsp2.hsp2.main import main

    io_manager = IOManager(filename)

    SaveLevel = True
    if "Light" in filetype:
        SaveLevel = False
    main(io_manager, saveall=SaveLevel, jupyterlab=False)
    # main('test.h5', saveall=True)
