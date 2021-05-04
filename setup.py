from setuptools import setup, find_packages
import versioneer

name = "deep_cave"
version = versioneer.get_version()
release = versioneer.get_versions()["full-revisionid"]

try:
    # use sphinx only in contexts where sphinx and its dependencies have been installed beforehand.
    from sphinx.setup_command import BuildDoc

    cmdclass = {"build_sphinx": BuildDoc}
    cmd_options = {
        "build_sphinx": {
            "project": ("setup.py", name),
            "version": ("setup.py", version),
            "release": ("setup.py", release),
            "source_dir": ("setup.py", "docs"),
            "build_dir": ("setup.py", "build"),
        }
    }
    kwargs = dict(command_options=cmd_options)
except ImportError as e:
    cmdclass = None
    cmd_options = None
    kwargs = {}


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    dependency_links=[
        "git+https://github.com/automl/fanova.git@2442ef36856a98f8755089f915ffe8373d9b5046#egg=fanova"
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.1",
            "more-itertools>=8.5.0",
            "tqdm>=4.56.0",
            "pipenv-setup>=3.1.1",
            "tox>=3.21.0",
            "coverage>=5.3.1",
            "pytest-cov>=2.10.1",
            "sphinx>=3.4.3",
            "recommonmark>=0.7.1",
            "versioneer>=0.19",
            "skl2onnx>=1.7.0",
            "scikit-optimize>=0.8.1",
        ]
    },
    name=name,  # Replace with your own username
    version=version,
    cmdclass=versioneer.get_cmdclass(cmdclass),
    author="Niels Nuthmann",
    author_email="niels.nuthmann@stud.uni-hannover.de",
    description="Automatic Analysis of Highly Efficient AutoML Tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nielsmitie/DeepCAVE",
    packages=find_packages(),
    classifiers=[],
    install_requires=[
        "pandas>=1.2.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.3",
        "dash>=1.18.1",
        "pyyaml>=5.3.1",
        "scipy>=1.6",
        "dash-dangerously-set-inner-html>=0.0.2",
        "onnxruntime>=1.6.0",
        "configspace>=0.4.17",
        "pdpbox>=0.2.1",
        "pyimp>=1.1.2",
        "dash-bootstrap-components>=0.11.1",
        "scikit-learn>=0.24",
    ],
    python_requires=">=3.8",
    entry_points={"console_scripts": ["server=deep_cave.server.__init__:main"]},
    **kwargs
)
