# HSP2, Hydrologic Simulation Program Python (HSPsquared)

HSPsquared or HSP2 is a Python version of [Hydrological Simulation Program - FORTRAN (HSPF)](https://www.epa.gov/ceam/hydrological-simulation-program-fortran-hspf).
HSP2 currently supports all HSPF hydrology modules and major water quality modules. Support for specialty modules is currently in progress. See our [Release Notes](https://github.com/respec/HSPsquared/releases) for up-to-date details.

Read our wiki for more information on our motivation and goals for HSP2:
- [Wiki Home & HSP2 Goals](https://github.com/respec/HSPsquared/wiki)
- [About-HSPF](https://github.com/respec/HSPsquared/wiki/About-HSPF)
- [Why-HSP2?](https://github.com/respec/HSPsquared/wiki/Why-HSP2%3F)

[Project slides (January 2017)](https://github.com/respec/HSPsquared/blob/archivePy2/Why%20HSP2%20(EAA).pdf) also provide helpful background.

HSPsquared is copyrighted 2020 by RESPEC and released under the GNU Affero General Public License.


## Repository Directories

**[HSP2](HSP2)** contains the hydrology and water quality code modules converted from HSPF, along with the main programs to run HSP2.

**[HSP2notebooks](HSP2notebooks)** contains tutorials and useful Juptyer Notebooks.

**[HSP2tools](HSP2tools)** contains supporting software modules such as the code to convert legacy WDM and UCI files to HDF5 files for HSP2, and to provide additional new and legacy capabilities.

**[docs](docs)** contains relevant reference documentation.

**[tests](tests)** contains unit testing code for testing code conversion (`tests/convert/conversion_test.py`) and code performance.


## Installation with Anaconda

HSP2 is designed to work with Python >3.6 (since April 2020). We **recommend Python 3.8**. Legacy Python 2 code is available in our [`archivePy2`](https://github.com/respec/HSPsquared/tree/archivePy2) branch.

Follow these steps to install using Anaconda.

#### 1. Install the Anaconda Python Distribution

We recommend installing the [latest release](https://docs.anaconda.com/anaconda/reference/release-notes/) of [**Anaconda Individual Edition**](https://www.anaconda.com/distribution). Follow their [installation](https://docs.anaconda.com/anaconda/install/) documentation.

#### 2. Clone or Download this HSPsquared repository

From this Github site, click on the green "Code" dropdown button near the upper right. Select to either Open in GitHub Desktop (i.e. git clone) or "Download ZIP". We recommend using GitHub Desktop, to most easily receive updates.

Place your copy of the HSPsquared folder in any convenient location on your computer.

#### 3. Create a Conda Environment for HSP2 Modeling (optional)

Although HSP2 can be run from the default `base` environment created by Anaconda,
it can be helpful to create a leaner custom environment.

We have provided an [`environment.yml`](environment.yml) file, which lists all primary dependencies, to help. Create a `hsp2_py38` environment either with the **Import** button on [Anaconda Navigator's Environments tab](https://docs.anaconda.com/anaconda/navigator/overview/#environments-tab), or use this [Conda](https://conda.io/docs/) command in your terminal or console,  replacing `path/environment.yml` with the full file pathway to the `environment.yml` file in the local cloned repository.

```shell
conda env create --file path/environment.yml
```
To update your environment, either use Anaconda Navigator, or run the following command:

```shell
conda env update --file path/environment.yml --prune
```

or

```shell
conda env create --file path/environment.yml --force
```

NOTE: The [`environment_dev.yml`](environment_dev.yml) file provides an alternate environment that provides additional capabilities and newer libraries useful to the development team. It is tested to also work with the current HSP2 codebase and will likely serve as a preview of future updates to [`environment.yml`](environment.yml).


#### 4. Add your HSPsquared Path to Anaconda sites-packages

To have access to the `HSP2` and `HSP2tools` modules in your Python environments,
it is necessary to have a path to your copy of HSPsquared in Anaconda's `sites-packages` directory (i.e. something like `$HOME/path/to/anaconda/lib/pythonX.X/site-packages` or `$HOME/path/to/anaconda/lib/site-packages` similar).

The easiest way to do this is to use the [conda develop](https://docs.conda.io/projects/conda-build/en/latest/resources/commands/conda-develop.html) command in the console or terminal like this, replacing `/path/to/module/` with the full file pathway to the local cloned HSPsquared repository:

```console
conda-develop /path/to/module/
```

You should now be able to run the Tutorials and create your own Jupyter Notebooks!



## Installation with Pip

Follow these steps to install using Pip. 

#### 1. Install Python and Pip

HSP2 is designed to work with Python >3.6 (since April 2020). We **recommend Python 3.8**. Legacy Python 2 code is available in our [`archivePy2`](https://github.com/respec/HSPsquared/tree/archivePy2) branch.

Instructions for downloading Python to your computer based on your operating system can be found in [this helpful wiki](https://wiki.python.org/moin/BeginnersGuide/Download).

`pip` is the [package installer for Python](https://packaging.python.org/en/latest/guides/tool-recommendations/). Check to see if `pip` is installed by running the following in the command line:

```shell
pip help
```

If you get an error, you will need to [install pip](https://pip.pypa.io/en/stable/installation/). Otherwise, both Python and pip are on your machine. 

#### 2. Clone or Download this HSPsquared repository

From this Github site, click on the green "Code" dropdown button near the upper right. Select to either Open in GitHub Desktop (i.e. git clone) or "Download ZIP". We recommend using GitHub Desktop, to most easily receive updates.

Place your copy of the HSPsquared folder in any convenient location on your computer.


#### 3. Create a new Environment for HSP2 Modeling (optional)

Although HSP2 can be run from the default environment in Python,
it can be helpful to create a leaner custom environment. The [`venv`]([venv — Creation of virtual environments — Python 3.9.9 documentation](https://docs.python.org/3.9/library/venv.html)) package can be used to manage Python environments. The documentation linked above will show you how to create and activate a new environment for running HSP2. 

#### 4. Pip install HSP2 

Navigate to your copy of the HSPsquared folder on your computer in the command line.

To install from the current local directory using pip:

```shell
pip install .
```

#### 5. Run HSP2 from the Command Line

The pip installed 'hsp2' command has help created from the function docstrings in HSP2tools/HSP2_CLI.py.

Use the help to learn how to use the model and each sub-command:

```shell
hsp2 --help
```
```shell
hsp2 import_uci --help
```
```shell
hsp2 run --help
```

Intended workflow from the command line:
```
hsp2 import_uci import_test.uci new_model.h5
hsp2 run new_model.h5
```




## Getting Started

We recommend looking over our [Understanding HSP2 Tutorial](HSP2notebooks/Tutorial1.md) then viewing or interactively running our [Introduction to HSP2 notebook](HSP2notebooks/Introduction.ipynb) Jupyter Notebook.

We recommend using [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) to run our tutorial [Juptyer Notebooks](https://jupyter.org/index.html) in the [HSP2notebooks](HSP2notebooks/) folder, due to many additional built-in features and extensions. The following JupyterLab [extensions](https://jupyterlab.readthedocs.io/en/stable/user/extensions.html) are particularly useful:
- [lckr-jupyterlab-variableinspector](https://github.com/lckr/jupyterlab-variableInspector)
