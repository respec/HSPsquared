import os
import re
import shlex
import sys
from setuptools import setup

exec(open('HSP2/_version.py').read())

if sys.argv[-1] == "publish":
    os.system(shlex.quote("cleanpy ."))
    os.system(shlex.quote("python setup.py sdist"))
    os.system(shlex.quote(f"twine upload dist/HSPsquared-{__version__}.tar.gz"))
    sys.exit()

README = open("./README.md").read()

pypi_map = {"jupyter-lsp-python": "python-lsp-server[all]"}


def process_env_yaml(fname, dev=False):
    yaml_lines = []
    with open("environment.yml") as fp:
        collect = False
        for line in fp.readlines():
            # Handle comments
            line = line.split("#")[0].rstrip()

            # Handle blank lines
            if not line.strip():
                continue

            line = line.strip(" -").replace(" ", "")

            if collect is True:
                words = re.split("=|<|>", line)
                # Shouldn't have interactivity, development tools, conda, pip,
                # or python as a dependency.
                if dev is True:
                    if words[0] in ["conda", "conda-build", "pip", "python", "pip:"]:
                        continue
                else:
                    if words[0] in [
                        "jupyterlab",
                        "ipywidgets",
                        "matplotlib",
                        "conda",
                        "conda-build",
                        "pip",
                        "python",
                        "lckr-jupyterlab-variableinspector",
                        "jupyter-lsp-python",
                        "jupyterlab-lsp",
                        "pip:",
                    ]:
                        continue

                # On PyPI pytables is tables.
                if words[0] == "pytables":
                    line = line.replace("pytables", "tables")

                yaml_lines.append(line)

            if line.rstrip() == "dependencies:":
                collect = True
    yaml_lines = [pypi_map.get(i, i) for i in yaml_lines]
    return yaml_lines


install_requires = process_env_yaml("environment.yml")
print(install_requires)
# extras_require = {
#     "dev": process_env_yaml("environment_dev.yml", dev=True) + ["cleanpy", "twine"]
# }

setup(
    name="HSPsquared",
    version=__version__,
    description="Hydrological Simulation Program - Python",
    long_description=README,
    classifiers=[
        # Get strings from
        # http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "Environment :: Console",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="hydrology hydrological hydraulic hspf",
    author="RESPEC, Inc",
    author_email="",
    url="http://www.respec.com/product/hydrologic-simulation-program-python-hsp%C2%B2/",
    packages=["HSP2", "HSP2tools", "HSP2IO"],
    include_package_data=True,
    package_data={"HSP2tools": ["data/*"]},
    zip_safe=False,
    install_requires=install_requires,
    # extras_require=extras_require,
    entry_points={"console_scripts": ["hsp2=HSP2tools.HSP2_CLI:main"]},
    test_suite="tests",
    python_requires=">3.6",
)
