BOHB
-----

To load an BOHB run into DeepCAVE, it is necessary to install HpBandSter, e.g. via:

   .. code-block:: bash

        pip install deepcave[bohb]

To be able to load it correctly into DeepCAVE, you need save the configuration space as a ConfigSpace JSON file:

   .. code-block:: python

      "configspace.json"