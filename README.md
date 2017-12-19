# HSP2, Hydrologic Simulation Program Python (HSPsquared)

HSPsquared or HSP2 is a Python version of HSPF. Currently it supports the major hydrology modules.  It is copyrighted by RESPEC and released under the GNU Affero General Public License.

## MAJOR REPOSITORY DIRECTORIES:

**HSP2** contains the hydrology codes converted from HSPF and the main programs to run HSP2. This software is Python with dependency on Pandas and Numba (open source) libraries.

**HSP2notebooks** contains tutorials and useful Juptyer Notebooks.  Some of the tutorials demonstrate capabilities that require additional Python modules (like Networkx and matplotlib.)

**HSP2tools** contains supporting software modules such as the code to convert legacy WDM and UCI files to HDF5 files for HSP2, and to provide additional new and legacy capabilities.

The remaining directories are versions of Tim Cera's open source code (wdmtoolbox, tstoolbox, and hspfbintoolbox) with versions tested to work with HSP2 and other open source code required by Tim Cera's modules.  It is expected that these will not be needed in the future when current versions have been proven to work.


## INSTALLATION INSTRUCTIONS

Install Python 2.7 and the additional scientific, open source libraries.  Ananconda (formerly Continuum Analytics) and Enthought provide free, simple installation of all the required Python and scientific libraries.  Check that numpy, Pandas, matplotlib, and numba are loaded.  Qgrid (version > 1.0) is also used in some of the Jupyter notebook tutorials.  Qgrid is a Jupyter notebook widget which uses SlickGrid to render pandas DataFrames within a Jupyter notebook. This allows you to explore your DataFrames with intuitive scrolling, sorting, and filtering controls, as well as edit your DataFrames by double clicking cells.The package managers included with Anaconda or Enthought's downloads makes this easy to check and add any missing packages.

From this Github site, click on the green "Clone or Download" dropdown button to select "Download ZIP".  Unzip the downloaded zipfile (in your Windows "Downloads" directory) and move it  to a convenient location such as your Desktop. You should now be able to run the Tutorials and create your own Jupyter Notebooks in this directory.

IF AND ONLY IF you need to run HSP2 outside of the distribution directory, you will need to use the Windows utility to edit your environmental variables. Find or create the PYTHONPATH environmental variable and add the path to where you placed the unzipped HSP2 distribution - remember to add the final backslash to the path. For example: "C:\Users\myusername\Desktop\HSPsquared\". 


## TUTORIALS and JUPYTER NOTEBOOKS

You should be able to start the Tutorials and other Jupyter Notebooks once you have finished the installation steps above.  In Enthought's Canopy distribution you can simply click on the desired file - but this is amazingly slow since it starts Canopy which in turn eventually starts the Notebook.  The easiest way with either Anaconda or Enthought is to open a command window, move (CD) to the location where you put the HSP2 unzipped file, and then type "jupyter notebook" at the command prompt.  You will see the Jupyter Notebook open a file browser window. Click on the desired Tutorial.  If you are using the Enthought Python distribution, Canopy, look for "Enthought Canopy" under the Windows "All Programs" to find the "Enthought Canopy Command Window" to use.  You can pin this to either your task bar or start window to make starting Notebooks easy.

There is also a YouTube video available at https://youtu.be/aeLScKsP1Wk to get you inroduced to HSP2.

NOTE: As a Jupyter project security step, the first time you start any Jupyter Notebook, you may need to look at this message in the command window: "Copy/past this  URL into your browser when you connect for the first time, to login with a token:".  You should copy and paste the following line into your browser.  You will NOT need to do this again.  The Jupyter system wants to insure that you are authorizing the Jupyter server to run on your system.  This is in rapid change to a password based authorization, so follow the instructions in the command window.
