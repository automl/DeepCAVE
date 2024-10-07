SMAC
----

For the SMAC framework, we support loading runs created with the older SMAC version 1.4 as well as
runs created with the newer SMAC version >= 2.0.0.

As Auto-Sklearn and Auto-PyTorch rely on SMAC for hyperparameter optimization, the SMAC
converter also allows to load runs created with these tools.

To be able to load it correctly into DeepCAVE, you need save the configuration space as a ConfigSpace JSON file,
as well as the run history:

   .. code-block:: python

      "configspace.json"
      "runhistory.json"

