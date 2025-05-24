RayTune
-------

To be able to load your RayTune run into DeepCave, there are a few points to
consider when running Raytune:

1. To load a RayTune run into DeepCAVE, it is necessary to install ray[tune], e.g. via:

   .. code-block:: bash

        pip install deepcave[raytune]

2. If there is not a configspace.json file provided it will be autoextracted. 
   For more reliable results please provide your own configspace.json file or 
   ajust the one provided. Numeric values will be treated as uniform values.
   Please also check if the objectives bounds as well as its goal are as wanted.
  
3. It is necessary to include the experiment json file provided in RayTune output.


.. note::
   There has to be exactly one experiment json file.

4. You can add a file called budget.json if you want to specify what is used as budget. 
   The content of the file should just be one string containing the budget name and needs
   to be placed next to the experiment json file. 
   

.. note::
   If no explicit budget is given, the default budget is set to the number of trials.

