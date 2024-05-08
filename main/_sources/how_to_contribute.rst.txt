==================
How to Contribute
==================
We would appreciate your help in making this project better.
Whether it's a bug report, new feature, correction, or additional documentation, we greatly value feedback and contributions from our community.
If you think you have found a bug, or have a new feature idea, please check for `github issues <github_issues_>`_ and create a new issue if needed.
If you'd like to work on a bug or new feature idea, or have a correction or addition to the documentation, please open a new issue and setup a development workflow to work on the issue and submit a pull request.

.. _contribute_fork_project:

Setup Development Workflow
---------------------------
The development workflow is based on the "fork-and-pull" Git workflow. A full description of the workflow can be found `here <fork_and_pull_>`_. To contribute to the project, follow these steps:

1. Fork the project.
2. Install the project in development mode as described in the :ref:`installation` section.
3. Create a new branch for the new feature on your own fork, note that all changes should be based on the `development` branch.
4. Make your changes and commit them to your branch.
5. Push your changes to your fork.
6. Open a pull request from your fork to the main repository's `development` branch.



.. _contribute_unit_tests:

Extend Unit Tests
------------------
When adding new features or making changes to the codebase, it is important to add unit tests and check for mistakes revealed by pre-existing test cases. This project uses `unittest` for testing.
To run the test suite, you can use the following command:

.. code-block:: bash

    python -m unittest discover -s tests


.. _contribute_update_documentation:

Update Documentation
---------------------

The documentation is written in reStructuredText and uses Sphinx to generate HTML documentation.
To build the documentation, you can use the following command:

.. code-block:: bash

    make docs


.. _contribute_pre_commit_hook:

Pre-Commit Hook
---------------
Before submitting a pull request, you should run the pre-commit hook to check for formatting errors and run the test suite, e.g. via

.. code-block:: bash

    pre-commit run --all-files

.. _contribute_pull_request:

Create Pull Request
--------------------

After making changes to the codebase, you can submit a pull request. Your pull request will be reviewed by the maintainers, and if everything looks good, it will be merged.
Before submitting a pull request, please make sure of the following:

- Make sure to reference the issue in the pull request.
- The pull request should include tests for the new feature or bug fix.
- The pull request should include documentation for the new feature or bug fix.
- All tests should pass.
- Make sure to select the correct source and destination repositories and according branches, usually your fork to the main repository's `development` branch.
- Add the change to the CHANGELOG.md file.
- The pull request should include a description of the change and the reason for the change.


.. _github_issues: https://github.com/automl/DeepCAVE/issues
.. _fork_and_pull: https://reflectoring.io/github-fork-and-pull/