Installation
============

DeepCAVE needs `redis-server <https://flaviocopes.com/redis-installation/>`_ to be installed
for the interactive mode, which can be done via:

.. code:: bash

    brew install redis  # MacOS
    sudo apt-get install redis-server  # Linux

The following commands install DeepCAVE. We recommend using anaconda, as this way `swig` can be
installed directly. If you use a different environment, make sure that
`swig <https://www.swig.org/index.html>`_ is installed.

.. code:: bash

    conda create -n DeepCAVE python=3.9
    conda activate DeepCAVE
    conda install -c anaconda swig
    pip install DeepCAVE

To load runs created with Optuna or the BOHB optimizer, you need to install the
respective packages by running:

.. code:: bash

    pip install deepcave[optuna]
    pip install deepcave[bohb]

To try the examples for recording your results in DeepCAVE format, run this after installing:

.. code:: bash

    pip install deepcave[examples]

If you want to contribute to DeepCAVE, you can clone it from GitHub and install the dev package:

.. code:: bash

    git clone https://github.com/automl/DeepCAVE.git
    conda create -n DeepCAVE python=3.9
    conda activate DeepCAVE
    conda install -c anaconda swig
    make install-dev


.. warning::

    DeepCAVE is officially tested and supported on Linux platforms.

    While it is generally expected to function correctly on MacOS, some issues may arise due to
    compatibility with Swig. Specifically, users may encounter problems with the
    Parallel Coordinates and Importance Plugin on MacOS.

    Currently, DeepCAVE cannot be run on Windows due to its reliance on a bash script for
    starting services such as Redis, workers, and the webserver.


Mac Related
^^^^^^^^^^^
If you want to run DeepCAVE on a M1 Mac, you need to add

.. code:: bash

    export DISABLE_SPRING=true
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES


to your ``~/.bash_profile`` to enable multi-processing.


Redis Server
^^^^^^^^^^^^

If you have problems installing `redis-server`, please try the following steps:

1. First, check if `redis-server` is available:

    .. code:: bash

        redis-server

2. If you see something like ``/usr/sbin/redis-server``, then you simply have to expand your path:

    .. code:: bash

        export PATH=$PATH:/usr/sbin

    Consider adding this to your ``~/.bashrc`` file.
    Check if `redis-server` works now.

3. If no `redis-server` was found, try to install it:

    .. code:: bash

        sudo apt-get install redis-server  # Linux
        brew install redis  # Mac

4. If there was no `redis-server` found and you do not have admin access, do the following inside the DeepCAVE root folder:

    .. code:: bash

        file="redis-6.2.6"
        filename=$file".tar.gz"

        mkdir -p vendors
        cd vendors
        wget https://download.redis.io/releases/$filename
        tar -xzvf $filename
        rm $filename
        cd $file
        make
        make install
        export PATH=$PATH:`pwd`
        cd ../../

