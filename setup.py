import sys
from codecs import open  # To use a consistent encoding
from os import path

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))
install_requirements = [
    "numpy",
    "python-dateutil",
    "spacepy",
    "cdflib>=1.3.3",
    "scipy",
    "matplotlib",
    "spiceypy",
    "requests",
    "bs4",
    "html5lib"
]

# The following are meant to avoid accidental upload/registration of this
# package in the Python Package Index (PyPi)
pypi_operations = frozenset(["register", "upload"]) & frozenset([x.lower() for x in sys.argv])
if pypi_operations:
    raise ValueError("Command(s) {} disabled in this example.".format(", ".join(pypi_operations)))

# Python favors using README.rst files (as opposed to README.md files)
# If you wish to use README.md, you must add the following line to your MANIFEST.in file::
#
#     include README.md
#
# then you can change the README.rst to README.md below.
with open(path.join(here, "README.rst"), encoding="utf-8") as fh:
    long_description = fh.read()

# We separate the version into a separate file so we can let people
# import everything in their __init__.py without causing ImportError.
__version__ = None
exec(open("mavenpy/about.py").read())
if __version__ is None:
    raise IOError("about.py in project lacks __version__!")

setup(
    name="mavenpy",
    version=__version__,
    author="Rebecca Jolitz",
    description="Routines to import and download MAVEN data.",
    long_description=long_description,
    license="BSD",
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    include_package_data=True,
    setup_requires=["numpy"],
    install_requires=install_requirements,
    keywords=["maven", "nasa", "swia", "sep", "mag", "swea"],
    url="https://bitbucket.org/rdjolitz/mavenpy",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
