Home
====

.. toctree::
   :hidden:
   :maxdepth: 2

   installation
   getting_started
   examples/index
   plugins/index
   converters
   api
   glossary
   faq


DeepCAVE is a visualization and analysis tool for AutoML (especially for the sub-problem
hyperparameter optimization) runs. The framework is programmed on top of Dash and therefore
entirely interactive. Multiple and diverse plugins make it possible to efficiently generate insights
and bring the human back in the loop. Moreover, the powerful run interface and the modularized
plugin structure allow extending the tool at any time effortlessly.


.. image:: ./images/plugins/configuration_footprint.png


Features
^^^^^^^^

- Interactive Dashboard (completely written in Python) to self-analyze optimization runs/processes.
- Analyzing while optimizing (run changes are automatically detected).
- A large collection of plugins to explore multiple areas like performance, hyperparameter and
  budget analysis.
- Save your runs using DeepCAVE's native recorder.
- Support for many optimizers using converter (e.g., DeepCAVE, SMAC and BOHB).
- Select runs directly from a working directory in the interface.
- Select groups of runs for combined analysis.
- Modularized plugin structure with access to selected runs/groups to provide maximal flexibility.
- Asynchronous execution of expensive plugins and caching of their results.
- Help buttons and integrated documentation in the interface helps you to understand the plugins.
- Use the matplotlib mode to customize and save the plots for your publication.
- The API mode gives you full access to the code, while you do not have to interact with the 
  interface. Otherwise, you can also make use of the raw data, provided by every plugin.



Citation
^^^^^^^^

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

