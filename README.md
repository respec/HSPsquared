# HSP2, Hydrologic Simulation Program Python (HSPsquared)

HSPsquared or HSP2 is a Python version of [Hydrological Simulation Program - FORTRAN (HSPF)](https://www.epa.gov/ceam/hydrological-simulation-program-fortran-hspf).
Currently it supports the major hydrology modules, and water quality modules are
being developed.

Read our wiki for more information on our motivation and goals for HSP2:
- [Wiki Home & HSP2 Goals](https://github.com/respec/HSPsquared/wiki)
- [About-HSPF](https://github.com/respec/HSPsquared/wiki/About-HSPF)
- [Why-HSP2?](https://github.com/respec/HSPsquared/wiki/Why-HSP2%3F)

[Project slides (January 2017)](https://github.com/respec/HSPsquared/blob/archivePy2/Why%20HSP2%20(EAA).pdf) also provide helpful background.

HSPsquared is copyrighted 2020 by RESPEC and released under the GNU Affero General
Public License.


**ANNOUNCEMENT**

HSP2 code has been updated to Python 3 in April 2020. Legacy Python 2 code is available in our [`archivePy2`](https://github.com/respec/HSPsquared/tree/archivePy2) branch.


## Repository Directories

**HSP2** contains the hydrology codes converted from HSPF and the main programs
to run HSP2.

**HSP2notebooks** contains tutorials and useful Juptyer Notebooks.

**HSP2tools** contains supporting software modules such as the code to convert
legacy WDM and UCI files to HDF5 files for HSP2, and to provide additional new
and legacy capabilities.


## Installation Instructions

HSP2 is designed to work with Python 3.6, 3.7 and 3.8.

Follow these steps to install.

#### 1. Install the Anaconda Python Distribution

We recommend installing the [latest release](https://docs.anaconda.com/anaconda/reference/release-notes/) of [**Anaconda Individual Edition**](https://www.anaconda.com/distribution). Follow their [installation](https://docs.anaconda.com/anaconda/install/) documentation.

#### 2. Clone or Download this HSPsquared repository

From this Github site, click on the green "Code" dropdown button near the upper right. Select to either Open in GitHub Desktop (i.e. git clone) or "Download ZIP". We recommend using GitHub Desktop, to most easily receive updates.

Place your copy of the HSPsquared folder in any convenient location on your computer.

#### 3. Create a Conda Environment for HSP2 Modeling (optional)

Although HSP2 can be run from the default `base` environment created by Anaconda,
it can be helpful to create a leaner custom environment.

We have provided an [`environment.yml`](environment.yml) file, which lists all primary dependencies, to help. Create a `hsp2_py37` environment either with the **Import** button on [Anaconda Navigator's Environments tab](https://docs.anaconda.com/anaconda/navigator/overview/#environments-tab), or use this [Conda](https://conda.io/docs/) command in your terminal or console,  replacing `path/environment.yml` with the full file pathway to the `environment.yml` file in the local cloned repository.

```console
conda env create --file path/environment.yml
```

NOTE: We recommend using [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) to run our tutorial [Juptyer](https://jupyter.org/index.html) Notebooks in the [HSP2notebooks](HSP2notebooks/) folder. The following JupyterLab [extensions](https://jupyterlab.readthedocs.io/en/stable/user/extensions.html) are useful (but not required):
+ [jupyter-matplotlib](https://github.com/matplotlib/ipympl#readme), , requires `nodejs`.
+ [`jupyterlab/toc`](https://github.com/jupyterlab/jupyterlab-toc), requires `nodejs`.
+ qgrid2



#### 4. Add your HSPsquared Path to Anaconda sites-packages

To have access to the `HSP2` and `HSP2tools` modules in your Python environments,
it is necessary to have a path to your copy of HSPsquared in Anaconda's `sites-packages` directory (i.e. something like `$HOME/path/to/anaconda/lib/pythonX.X/site-packages` or `$HOME/path/to/anaconda/lib/site-packages` similar).

The easiest way to do this is to use the [conda develop](https://docs.conda.io/projects/conda-build/en/latest/resources/commands/conda-develop.html) command in the console or terminal like this, replacing `/path/to/module/` with the full file pathway to the local cloned HSPsquared repository:

```console
conda-develop /path/to/module/
```

You should now be able to run the Tutorials and create your own Jupyter Notebooks!


## Getting Started

We recommend looking over our [Understanding HSP2 Tutorial](HSP2notebooks/Tutorial1.md) then viewing or interactively running our [Introduction to HSP2 notebook](HSP2notebooks/Introduction.ipynb).
