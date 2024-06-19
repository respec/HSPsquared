Hydrologic Simulation Program - Python (HSP2)
=============================================

The Hydrologic Simulation Program–Python (HSP2_) watershed model is is a port
of the well-established Hydrological Simulation Program - FORTRAN (HSPF_),
re-coded with modern scientific Python and data formats.

HSP2 (pronouced "HSP-squared") is being developed by an open source team
launched and led by RESPEC with internal funding. Our list of collaborators is
growing, now including LimnoTech and with additional support from the U.S. Army
Corps of Engineers (Engineer Research and Development Center (ERDC),
Environmental Laboratory), modelers at the Virginia Department of Environmental
Quality, and others.

HSP2 currently supports all HSPF hydrology and detailed water quality modules.
Support for specialty modules is currently in progress. See our `Release
Notes`_ for up-to-date details.

Read our wiki for more information on our motivation and goals for HSP2:
- `Wiki Home & HSP2 Goals <https://github.com/respec/HSPsquared/wiki>`_
- `About-HSPF <https://github.com/respec/HSPsquared/wiki/About-HSPF>`_
- `Why-HSP2? <https://github.com/respec/HSPsquared/wiki/Why-HSP2%3F>`_
- `HSP2 Design Details <https://github.com/respec/HSPsquared/wiki/HSP2_Design_Details>`_

The `HSPF Conversion Project`_ slides (January 2017) and the `Introduction to
HSP2 by Jason Love (RESPEC)`_ video (December 2017) provide additional
background on the `initial release`_.

HSPsquared is released under the `GNU Affero General Public License (AGPL),
copyrighted 2017 by RESPEC`_.

Source Code Directories
-----------------------

- **HSP2** contains the hydrology and water quality code modules converted from
  HSPF, along with the main programs to run HSP2.

- **HSP2tools** contains supporting software modules such as the code to
  convert legacy WDM and UCI files to HDF5 files for HSP2, and to provide
  additional new and legacy capabilities.

- **HSP2IO** is new in v0.10 and contains an abstracted approach to getting
  data in and out of HSP2 for flexibility and performance and also to support
  future automation and model coupling. - NOTE: With v0.10 the I/O abstraction
  classes provide an alternate approach to running HSP2. Our plan is to migrate
  solely using the I/O abstracted methods, but we will maintain both approaches
  for for several more releases for backward compatibility. 

- **docs** contains relevant reference documentation.

- **examples** contains examples of how to use HSP2, organized as interactive
  Jupyter Notebook tutorials.

- **tests** contains HSPF use cases, their input files, code to compare HSP2 vs
  HSPF model outputs and code to test for performance.

Getting Started
===============
We recommend getting started by:

1. Following our `HSP2 Installation`_ Instructions.

2. Opening our interactive JupyterLab_ HSP2 tutorials in the `examples`
   sub-directory.

HSP2 Installation
=================
We **recommend Python 3.10**. 

Install From Pre-built Packages
-------------------------------
Python Package Index (PyPI)
+++++++++++++++++++++++++++
Starting with version 0.11.0a1 we provide a PyPI wheel package for HSP2 which
should work on any supported platform for Python 3.10, 3.11, and 3.12.

.. code-block:: console

    python -m pip install hsp2

Windows Executable
++++++++++++++++++
On the Releases_ page, we provide a Windows package in the zip file named
HSP2_Driver_MonthYear.zip. HSP2_Driver_MonthYear.zip contains an .exe for
running HSP2, enabling a user to run HSP2 without needing to do anything with
Python code or Jupyter notebooks. The driver uses a file dialog to prompt for
the name of the HDF5 file to run, or if that doesn't exist yet you can give it
the name of a UCI or WDM file to import. It also runs with the H5 file name on
the command line.

Install From Source
-------------------
Clone or Download the HSPsquared Repository
+++++++++++++++++++++++++++++++++++++++++++
From the HSP2squared_ Github page, download and extract the code using one of
the options found by clicking on the green "Code" drop down button near the
upper right of the page, or by downloading one of the compressed source files
from the Releases_ page.

Place your copy of the HSPsquared folder in any convenient location on your
computer.

For the rest of the installation steps, let's call this location
`/path/to/module/hsp2`.

Create a Python Environment
+++++++++++++++++++++++++++
We provide two options to installing HSP2, yet recommend option 1.

Install using only one of these options.

Option 1: Install using "conda"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Follow these steps to install using the conda_ package manager.

1. Install the Anaconda Python Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Install the `latest release`_ of the Anaconda Distribution, which includes the
conda package manager, a complete Python (and R) data science stack, and the
Anaconda Navigator GUI.  Follow `Anaconda Installation`_ documentation.

A lighter-weight alternative is to install Miniconda_.

2. Create a Conda Environment for HSP2 Modeling (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Although HSP2 can be run from the default `base` environment created by
Anaconda, we recommend creating a custom environment that includes the exact
combination of software dependencies that we've in development and testing.

Use the following `conda create`_ command in your terminal or console:

.. code-block:: console

    conda create -c conda-forge -n hsp2_310 python=3.10 

Install the necessary and optional packages for HSP2 in the new environment:

.. code-block:: console

    conda install -c conda-forge -n hsp2_310 cltoolbox numba pandas pytables
    conda install -c conda-forge -n hsp2_310 h5py jupyterlab matplotlib notebook

.. code-block:: console

    conda activate hsp2_310

    cd /path/to/module/hsp2
    pip install .  # or "pip install -e ." to install in editable mode

You should now be able to run the Tutorials and create your own Jupyter
Notebooks!

Option 2: Install From Source Code Using `pip`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Installing HSP2 using `pip`, the `Package Installer for Python`_
is an alternative method to installing with `conda`. 

1. Install Python
^^^^^^^^^^^^^^^^^
Instructions for downloading Python to your computer based on your operating
system can be found in `this helpful wiki`_.

2. Create a Python Environment for HSP2 Modeling (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Create a custom Python virtual environment for using HSP2, following the `venv
— Creation of virtual environments`_ package documentation to create and
activate a new environment for running HSP2.

.. code-block:: console

    python -m venv hsp2_env /path/to/python/virtual/environments/hsp2_env

3. PIP install HSP2 
^^^^^^^^^^^^^^^^^^^
Navigate to your copy of the HSPsquared folder (for these instructions
/path/to/module/hsp2) on your computer in the command
line.

To install using pip:

.. code-block:: console

    source /path/to/python/virtual/environments/hsp2_env/bin/activate
    cd /path/to/module/hsp2
    pip install .  # or "pip install -e ." to install in editable mode

4. Run PIP Installed HSP2 from the Command Line
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The pip installed 'hsp2' command has help created from the function docstrings
in HSP2tools/HSP2_CLI.py.

Command Line Usage
++++++++++++++++++
Use the help to learn how to use the model and each sub-command:

.. code-block:: console

    hsp2 --help
    hsp2 import_uci --help
    hsp2 run --help

Intended workflow from the command line:

.. code-block:: console

    hsp2 import_uci import_test.uci new_model.h5
    hsp2 run new_model.h5

API Usage
+++++++++
The HSP2 API is designed to be used in Python scripts and Jupyter notebooks.

.. code-block:: python

    from HSP2 import HSP2

.. _HSP2: https://github.com/respec/HSPsquared
.. _Releases: https://github.com/respec/HSPsquared/releases
.. _HSPF: https://www.epa.gov/ceam/hydrological-simulation-program-fortran-hspf
.. _`Release Notes`: https://github.com/respec/HSPsquared/releases
.. _`HSPF Conversion Project`: https://github.com/respec/HSPsquared/blob/archivePy2/Why%20HSP2%20(EAA).pdf
.. _`Introduction to HSP2 by Jason Love (RESPEC)`: https://www.youtube.com/watch?v=aeLScKsP1Wk
.. _`initial release`: https://github.com/respec/HSPsquared/releases/tag/0.7.7
.. _`GNU Affero General Public License (AGPL), copyrighted 2017 by RESPEC`: https://github.com/respec/HSPsquared/blob/master/LICENSE 
.. _JupyterLab: https://jupyterlab.readthedocs.io/en/stable/
.. _conda: https://docs.conda.io/en/latest/
.. _`latest release`: https://docs.anaconda.com/anaconda/reference/release-notes/
.. _`Anaconda Installation`: https://docs.anaconda.com/anaconda/install/
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _HSP2squared: https://github.com/respec/HSPsquared
.. _`conda create`: https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-environments
.. _`conda develop`: https://docs.conda.io/projects/conda-build/en/latest/resources/commands/conda-develop.html
.. _`Package Installer for Python`: https://packaging.python.org/en/latest/guides/tool-recommendations/
.. _`this helpful wiki`: https://wiki.python.org/moin/BeginnersGuide/Download
.. _`venv — Creation of virtual environments`: https://docs.python.org/3.9/library/venv.html
