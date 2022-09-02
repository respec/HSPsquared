# Hydrologic Simulation Program - Python (HSP2)

The **[Hydrologic Simulation Program–Python (HSP2)](https://github.com/respec/HSPsquared)** watershed model is 
is a port of the well-established [Hydrological Simulation Program - FORTRAN (HSPF)](https://www.epa.gov/ceam/hydrological-simulation-program-fortran-hspf), re-coded with modern scientific Python and data formats.

HSP2 (pronouced "HSP-squared") is being developed by an open source team launched and led by RESPEC with internal funding and now in collaboration with LimnoTech and with additional support from the U.S. Army Corps of Engineers, Engineer Research and Development Center (ERDC), Environmental Laboratory.

HSP2 currently supports all HSPF hydrology and detailed water quality modules. Support for specialty modules is currently in progress. See our [Release Notes](https://github.com/respec/HSPsquared/releases) for up-to-date details.

Read our wiki for more information on our motivation and goals for HSP2:
- [Wiki Home & HSP2 Goals](https://github.com/respec/HSPsquared/wiki)
- [About-HSPF](https://github.com/respec/HSPsquared/wiki/About-HSPF)
- [Why-HSP2?](https://github.com/respec/HSPsquared/wiki/Why-HSP2%3F)
- [HSP2 Design Details](https://github.com/respec/HSPsquared/wiki/HSP2_Design_Details)

The [HSPF Conversion Project](https://github.com/respec/HSPsquared/blob/archivePy2/Why%20HSP2%20(EAA).pdf) slides (January 2017) and the [Introduction to HSP2 by Jason Love (RESPEC)](https://www.youtube.com/watch?v=aeLScKsP1Wk) video (December 2017) provide additional background on the [intial release](https://github.com/respec/HSPsquared/releases/tag/0.7.7).

HSPsquared is released under the [GNU Affero General Public License (AGPL), copyrighted 2017 by RESPEC](https://github.com/respec/HSPsquared/blob/master/LICENSE).


## Repository Directories

- **[HSP2](HSP2)** contains the hydrology and water quality code modules converted from HSPF, along with the main programs to run HSP2.

- **[HSP2tools](HSP2tools)** contains supporting software modules such as the code to convert legacy WDM and UCI files to HDF5 files for HSP2, and to provide additional new and legacy capabilities.

- **[HSP2IO](HSP2IO)** is new in v0.10 and contains an abstracted approach to getting data in and out of HSP2 for flexibility and performance and also to support future automation and model coupling. 
  - NOTE: With v0.10 the I/O abstraction classes provide an alternate approach to running HSP2. Our plan is to migrate solely using the I/O abstracted methods, but we will maintain both approaches for for several more releases for backward compability. 

- **[docs](docs)** contains relevant reference documentation.

- **[examples](examples)** contains examples of how to use HSP2, organized as interactive Juptyer Notebook tutorials.

- **[tests](tests)** contains HSPF use cases, their input files, code to compare HSP2 vs HSPF model outputs ([`tests/convert/conversion_test.py`](tests/convert/conversion_test.py)), and code to test for
 performance.


# Getting Started

We recommend getting started by:

1. Following our [HSP2 Installation](#HSP2-Installation) Instructions.

2.  Opening our [interactive HSP2 tutorials](examples) in [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/).  


# HSP2 Installation

HSP2 is designed to work with Python 3.7, 3.8, and 3.9. 

We presently **recommend Python 3.8**. 

We provide two options to installing HSP2, yet strongly recommend option 1.
1. [Install Option 1 using `conda`](#install-option-1-using-conda)
2. [Install Option 2 using `pip`](#install-option-2-using-pip)

Install using only one of these options.

## Install Option 1 using `conda`

Follow these steps to install using the [conda](https://docs.conda.io/en/latest/) package manager.

### 1. Install the Anaconda Python Distribution

We recommend installing the [latest release](https://docs.anaconda.com/anaconda/reference/release-notes/) of [**Anaconda Individual Edition**](https://www.anaconda.com/distribution), which includes the conda, a complete Python (and R) data science stack, and the helpful Anaconda Navigator GUI.
- Follow [Anaconda Installation](https://docs.anaconda.com/anaconda/install/) documentation.

A lighter-weight alternative is to install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### 2. Clone or Download this HSPsquared repository

From the [HSP2squared](https://github.com/respec/HSPsquared) Github page, click on the green "Code" dropdown button near the upper right. Select to either "Open in GitHub Desktop" (i.e. git clone) or "Download ZIP". We recommend using GitHub Desktop, to most easily receive updates.

Place your copy of the HSPsquared folder in any convenient location on your computer.

### 3. Create a Conda Environment for HSP2 Modeling (optional)

Although HSP2 can be run from the default `base` environment created by Anaconda, we recommend creating a custom environment that includes the exact combination of software dependencies that we've in development and testing.

Create the `hsp2_py38` environment from our [`environment.yml`](environment.yml) file, which lists all primary dependencies, using one of these approaches: 
1. Use the **Import** button on [Anaconda Navigator's Environments tab](https://docs.anaconda.com/anaconda/navigator/overview/#environments-tab), or 
2. Use the following [`conda create`](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-environments) command in your terminal or console,  replacing `path/environment.yml` with the full file pathway to the [`environment.yml`](environment.yml) file in the local cloned repository.

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


### 4. Add your HSPsquared Path to Anaconda sites-packages

To have access to the `HSP2`, `HSP2tools`, and `HSP2IO` modules in your Python environments, it is necessary to have a path to your copy of HSPsquared in Anaconda's `sites-packages` directory (i.e. something like `$HOME/path/to/anaconda/lib/pythonX.X/site-packages` or `$HOME/path/to/anaconda/lib/site-packages` similar).

- The easiest way to do this is to use the [conda develop](https://docs.conda.io/projects/conda-build/en/latest/resources/commands/conda-develop.html) command in the console or terminal like this, replacing `/path/to/module/` with the full file pathway to the local cloned HSPsquared repository:

    ```console
    conda-develop /path/to/module/
    ```

You should now be able to run the Tutorials and create your own Jupyter Notebooks!



## Install Option 2 using `pip`

Installing HSP2 using `pip`, the [Package Installer for Python](https://packaging.python.org/en/latest/guides/tool-recommendations/), is an alternative method to installing with `conda`. 

**WARNING**: If you followed [Install Option 1 using `conda`](#install-option-1-using-conda), then DO NOT also install using `pip`. Your install is complete and you can ignore all installatino steps below.


### 1. Install Python and Pip

Instructions for downloading Python to your computer based on your operating system can be found in [this helpful wiki](https://wiki.python.org/moin/BeginnersGuide/Download).

Check to see if `pip` is installed by running the following in the command line:

```shell
pip help
```

If you get an error, you will need to [install pip](https://pip.pypa.io/en/stable/installation/). Otherwise, both Python and pip are on your machine. 

### 2. Clone or Download this HSPsquared repository

Follow the instructions
 in [Install with Conda Step 2](#clone-or-download-this-hspsquared-repository), above.


### 3. Create a Python Environment for HSP2 Modeling (optional)

We strongly recommend creating custom Python virtual environment for using HSP2, following the [`venv` — Creation of virtual environments](https://docs.python.org/3.9/library/venv.html) package documentation to create and activate a new environment for running HSP2. 

### 4. PIP install HSP2 

Navigate to your copy of the HSPsquared folder on your computer in the command line.

To install from the current local directory using pip:

```shell
pip install .
```

### 5. Run HSP2 from the Command Line

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
