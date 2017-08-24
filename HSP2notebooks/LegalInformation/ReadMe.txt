Instructions for installation and use of files in this release package.

1. Install Canopy - the 64 bit version.  (Most other Python distributions that include Numba will work also.)   Make this your default Python distribution when prompted.

2. Login with your account (top of the Canopy start window.) This is necessary to see the packages not in the free verson. From the Canopy start window select the package manager.  Check to make sure the following are installed (click on them to install as necessary.)
   pytables
   h5py
   
3. Please read the License.txt file! Do not use HSP2 if you don't accept the license.

4. Double click on the HSP2 intaller which has a name like HSP-xxx.win-amd64.exe where xxx is the version number for HSP2.  This is a normal Windows installer, so continue to answer the questions until done.

5. The tutorials are designed to explain HSP2, demonstrate how to use various features, and provide additional information to help you take full advantage of HSP2. To start the tutorials, type "ipython notebook" in a Canopy Command Prompt window.  It will start the IPython Notebook manager in your default browser.  If you are using IE (Internet Explorer) with a version older than 10, you will need to upgrade. The tutorials look and operate better in Chrome or Firefox.

5. Start with Tutorial 1 and progress in order.  The tutorial and other notebooks provide example data and images in the Data directory. Please do not modify anything in this directory!  

The  data needed for the tutorials is copied to the ExampleData when the tutorial is started. This allows you
to safely play with the the examples without breaking the tutorials. Each time you restart, you make a fresh copy of the data.  So, feel free to modify the examples, play around, and try things to learn in more detail. You can't hurt anything.

Some Tutorials will need additional Python libraries that are NOT needed by the HSP2 code.  These are listed with the tutorials.  In most cases, you can use the Canopy Package Manager to install these.  If the module is not listed for the Package manager, then type 
		"pip install name" 
in a Canopy Command Prompt window to install it (name is the desired library name).  Packages installed this way do not show up in the Package Manager's display, but are properly installed and functional.  Later if you need to install an updated version of one of these packages, type 
		"pip install --upgrade name"
