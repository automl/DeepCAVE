.. _custom-converter:

================
Adding a Converter
================

Introduction
------------

This tutorial will guide you through the steps to add a new converter in DeepCAVE.
We will create a simple converter that demonstrates the basic structure and functionality.
By the end of this tutorial, you will have a better understanding of how to create and use converters in DeepCAVE.

Step 1: Create the converter class
----------------------------------

1. Start by creating a new Python file in the `deepcave/runs/converters` directory of your DeepCAVE installation.
   For example, you can name it my_first_converter.py.

2. Define the basic structure of your converter. You can start by using the following code:

    .. code-block:: python

        from deepcave.runs.run import Run

        class MyFirstConverter(Run):
            @property
            def hash(self) -> str:
                """
                Return a unique hash for the run (e.g., hashing the trial history).
                """
                pass

            @property
            def latest_change(self) -> float:
                """
                Return the timestamp of the latest change.
                """
                pass

            @classmethod
            def from_path(cls, path: str) -> 'Run':
                """
                Return a Run object from a given path.
                """
                pass
            
            @classmethod
            def is_valid_run(cls, path: str) -> bool:
                """
                Check if the path belongs to a valid run.
                """
                pass


Step 2: Register your converter
-------------------------------
After creating your new converter, you need to add it to the list of available converters in the `deepcave/config.py` file.
In addition, please add a short description to the `__init__.py` file in the `deepcave/runs/converters` directory.

The next time you select a run, DeepCAVE will automatically use this converter as long as the
`from_path` method did not fail.

Step 3: Implement the methods
-----------------------------
1. **hash**: This method should return a unique hash for the run. You might hash the trial history or
   use another unique identifier.
2. **latest_change**: This method should return the timestamp of the latest change.

3. **from_path**: This method should return a :ref:`Run<deepcave.runs.run>` object from a given path.
   Inside this method, you have to provide a configspace, objectives, and meta data, as well as the
   trials, which have to be added to the newly created run object.

4. **is_valid_run**: This method should return True if the given path belongs to a valid run. Valid means that the given run
   directory contains all files that are necessary for further processing (e.g. a configspace.json).
   
Step 5: Document your converter
-------------------------------

To document your converter,  create a corresponding `.rst` file under `docs/converters` and add it to
the `docs/converters/index.rst` file.
Additionally, define a help class variable in `my_first_converter.py` pointing to your documentation file:

.. code-block:: python

    # Example of help variable
    help = "docs/converters/my_first_converter.rst"

Step 5: Congratulate yourself
-----------------------------

Congratulations! You've just created your first DeepCAVE converter.
If you have any questions or need further assistance, feel free to reach out.