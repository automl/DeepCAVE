.. image:: ./images/DeepCAVE_Logo_wide.png
   :alt: DeepCAVE Logo
|

Home
====

.. toctree::
   :hidden:
   :maxdepth: 2

   installation
   getting_started
   examples/index
   plugins/index
   converters/index
   how_to_contribute
   api
   glossary
   faq


DeepCAVE is a visualization and analysis tool for AutoML, with a particular focus on
hyperparameter optimization (HPO). Built on the Dash framework, it offers a fully
interactive experience. The tool features a variety of plugins that enable efficient insight
generation, aiding in understanding and debugging the application of HPO.
Additionally, the powerful run interface and the modularized plugin structure allow extending the
tool at any time effortlessly.


.. image:: ./images/plugins/configuration_footprint.png


Features
--------

Interactive Interface
^^^^^^^^^^^^^^^^^^^^^

- **Interactive Dashboard:**
  The dashboard runs in a webbrowser and allows you to self-analyze your optimization runs interactively.

- **Run Selection Interface:**
  Easily select runs from your working directory directly within the interface.

- **Integrated Help and Documentation:**
  Use help buttons and integrated documentation within the interface to better understand the plugins.


Comprehensive Analysis Tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Extensive Plugin Collection:**
  Explore a wide range of plugins for in-depth performance, hyperparameter, and budget analysis.

- **Analysis of Running Processes:**
  Analyze and monitor optimization processes as they occur, with automatic detection of run changes.

- **Group Analysis:**
  Choose groups of runs for combined analysis to gain deeper insights.


Flexible and Modular Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Modular Plugin Architecture:**
  Benefit from a modularized plugin structure with access to selected runs and groups, offering you maximum flexibility.

- **Asynchronous Execution:**
  Utilize asynchronous execution of resource-intensive plugins and caching of results to improve performance.


Broad Optimizer Support
^^^^^^^^^^^^^^^^^^^^^^^

- **Optimizer Support:**
  Work with many frameworks and optimizers using our converters, including converters for SMAC, BOHB, AMLTK, and Optuna.

- **Native Format Saving:**
  Save AutoML runs from various frameworks in DeepCAVE's native format using the built-in recorder.

- **Flexible Data Loading:**
  Alternatively, load AutoML runs from other frameworks by converting them into a Pandas DataFrame.


Developer and API Features
^^^^^^^^^^^^^^^^^^^^^^^^^^

- **API Mode:**
  Interact with the code directly through API mode, allowing you to bypass the graphical interface if preferred.


Citation
--------

If you use DeepCAVE in one of your research projects, please cite our `ReALML@ICML'22 workshop paper <https://arxiv.org/abs/2206.03493>`_:

.. code-block:: text

    @misc{sass-realml2022,
      title = {DeepCAVE: An Interactive Analysis Tool for Automated Machine Learning},
      author = {Sass, René and Bergman, Eddie and Biedenkapp, André and Hutter, Frank and Lindauer, Marius},
      doi = {10.48550/ARXIV.2206.03493},
      url = {https://arxiv.org/abs/2206.03493},
      publisher = {arXiv},
      year = {2022},
      copyright = {arXiv.org perpetual, non-exclusive license}
    }

