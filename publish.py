"""
Publish package to PyPI.

This script is used to publish the package to PyPI. It is a simple script that
uses the `build` module to create the source distribution and wheel, and then
uses `twine` to upload the package to PyPI.

Need to install the following packages:
- build (if using PyPI), python-build (if using conda-forge)
- twine
"""

import shlex
import shutil
import subprocess

PKG_NAME = "hsp2"

with open("VERSION", encoding="ascii") as version_file:
    version = version_file.readline().strip()

shutil.rmtree("build", ignore_errors=True)

subprocess.run(shlex.split("python3 -m build --wheel"), check=True)
wheel = f"dist/{PKG_NAME}-{version}*.whl"

for file in [wheel]:
    subprocess.run(
        shlex.split(f"twine check {file}"),
        check=True,
    )
    subprocess.run(
        shlex.split(f"twine upload --skip-existing {file}"),
        check=True,
    )

shutil.rmtree("build", ignore_errors=True)
