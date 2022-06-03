Installation
============

DeepCAVE needs `redis-server <https://flaviocopes.com/redis-installation/>`_ for the interactive
mode.

.. code:: bash

    brew install redis  # Mac
    sudo apt-get install redis-server  # Linux


.. note:: 

    If you have problems see next section for extened instructions.


.. warning:: 

    DeepCAVE is tested on Linux and Mac only. Since a bash script is used to start the services
    (redis, workers and webserver), it is not possible to run DeepCAVE on Windows.


The following commands install DeepCAVE. We recommend using anaconda as `swig` can be installed
directly. If you use a different environment, make sure
`swig <https://www.dev2qa.com/how-to-install-swig-on-macos-linux-and-windows/>`_ is installed.

.. code:: bash

    conda create -n DeepCAVE python=3.9
    conda activate DeepCAVE
    conda install -c anaconda swig
    pip install DeepCAVE


If you want to contribute to DeepCAVE get it from GitHub directly and install the dev package:

.. code:: bash

    git clone https://github.com/automl/DeepCAVE.git
    conda create -n DeepCAVE python=3.9
    conda activate DeepCAVE
    conda install -c anaconda swig
    make install-dev



Redis Server
^^^^^^^^^^^^

If you have problems installing `redis-server` try the following steps:

1. First check if `redis-server` is available:

.. code:: bash

    redis-server

2. If you see something like `/usr/sbin/redis-server`, then you simply have to expand your path:

.. code:: bash

    export PATH=$PATH:/usr/sbin

Consider adding this to your `~/.bashrc` file.
Check if `redis-server` works now.

3. If no `redis-server` was found, try to install it:

.. code:: bash

    sudo apt-get install redis-server  # Linux
    brew install redis  # Mac

4. If there was no `redis-server` found and you do not have admin access,
do the following inside the DeepCAVE root folder:

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
    export PATH=$PATH:`pwd`
    cd ../../



Mac Related
^^^^^^^^^^^
If you are on a M1 Mac you have to add

.. code:: bash

    export DISABLE_SPRING=true
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES


to your ```~/.bash_profile``` to enable multi-processing.
