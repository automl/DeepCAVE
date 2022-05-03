Install Redis Server
====================

`redis-server` is required to make DeepCAVE run in interactive mode. This page gives information
how to get `redis-server` installed.


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

    sudo apt-get install redis-server  # linux
    brew install redis  # mac


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