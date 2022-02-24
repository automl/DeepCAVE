Installation
============

First, make sure you have `redis-server <https://flaviocopes.com/redis-installation/>`_
installed on your computer. You also need `swig <https://www.dev2qa.com/how-to-install-swig-on-macos-linux-and-windows/>`_
which will be installed automatically if you install via conda.

If you are on an Non-Intel Mac you have to add

.. code:: bash

    export DISABLE_SPRING=true
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES


to your ```~/.bash_profile``` to enable multi-processing.

Afterwards, follow the instructions:

.. code:: bash

    git clone https://github.com/automl/DeepCAVE.git
    cd DeepCAVE
    conda env create -f environment.yml
    conda activate DeepCAVE
    make install


If you want to contribute to DeepCAVE also install the dev packages:

.. code:: bash

    make install-dev
