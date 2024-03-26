from setuptools import setup

version = (
    open("./_version.py").read().strip().replace("__version__ = ", "").replace("'", "")
)


README = open("./README.md").read()


setup(
    name="hsp2",
    version=version,
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
    py_modules=["_version"],
    include_package_data=True,
    package_data={"HSP2tools": ["data/*"]},
    zip_safe=False,
    install_requires=["numpy", "pandas", "numba"],
    # extras_require=extras_require,
    entry_points={"console_scripts": ["hsp2=HSP2tools.HSP2_CLI:main"]},
    test_suite="tests",
    python_requires=">3.6",
)
