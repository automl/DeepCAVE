# Version 1.3.4

## Bug-Fixes
- Fix dependency issues 

# Version 1.3.3

## Bug-Fixes
- Fixed configuration footprint for Mac platforms
- fix bug in plots

## Converters
- Add for each converter a method to check whether a run is valid.
- Change rst docs of converters.

## Quality of Life
- Improve documentation on installation for Mac platforms

## General Layout
- Fix navigation and button problems in general layout.
- Change general layout to be more intuitive.

# Version 1.3.2

## Features
- Allow Importance and ablation path analysis for multi-objective runs.

# Version 1.3.1

## Quality of Life
- Improve documentation on installation of extra requirements.
- Add licence information to all source files.

# Version 1.3

## Converters
- Add support for AMLTK.
- Add support for Optuna.
- Add support for adding a run in DataFrame format (#137).
- BOHB: Consider missing state as valid state as state is not provided by default (#65).

## Version-Updates
- ConfigSpace from 0.6.1 to 1.2.0
- Numpy from 1.26.4 to 2.0.1
- pyPDPPartitioner from to 0.1.8 to 0.1.9

## Ablation Paths Plugin
- Added ablation paths as a plugin.
- Added ablation as a evaluator to use for the plugin.
- Added tests as well as documentation.
- Modified the RandomForest surrogate model, so it can be passed the number of trees.

## Move plugins
- Moved the Configuration cube and the parallel coordinates plugin from objective to hyperparameter.
- Changed layout accordingly as well as the documentation.

## Bug-Fixes
- Upgrade pyPDPPartitioner to 0.1.9 as a requirement which fixes a bug related to copying config space objects (#111).

## Quality of Life
- Update the documentation to reflect the new changes.
- Disable warnings during RunHandler initialization to avoid showing warnings with respect to previously loaded runs.
- Remove option to use docker (not yet supported).
- Add SMAC3v2 multi-fidelity example runs.
- Add more useful SMAC3v2 multi-objective example run.
- Remove defective matplotlib button and associated functionalities.

# Version 1.2.1

## Quality of Life
- Runs now get displayed with their parent directory for better distinguishability.
- Increase plot font sizes.
- Add a simple loading bar functionality for longer runs.

## General
- Seed is now required in the Recorder.

## Bug-Fixes
- Use normalized LPI importance via variance instead of importance over mean (#152)
- Return nan as importance values if variance is 0. for a hyperparameter / budget (#152)

## Plugins
- Show a run's hoover-text for the actual budget of a trial in Cost over Time with Combined budget (#154).
- Use highest budget as default budget for Cost over Time instead of Combined.
- Show best value / config for each objective instead of merged objective in Overview (#159).
- Use chosen objective instead of merged objective to get the incumbent for the calculation of LPI importance (#159).
- Add total runtime in overview (#155).

# Version 1.2

## Plugins
- Add symbolic explanations plugin (#46).
- It is now possible to view multiple unequal runs at once in Cost over Time and Pareto (#93).
- Runs with unequal objectives cannot be displayed together.
- Added an enum for displaying according warning messages.

## Enhancements
- Fix lower bounds of dependency versions.
- Allow to load multi-objective SMAC3v2 and add example (#69).
- Allow to load runs with multiple seeds and add examples (#70).
- Correct incumbent calculation when single objective should be maximized.
- Correct range of configuration cube slider for number of configs.
- Do not disable existing loggers.
- Update author email.
- Add exit button which first deletes running jobs and then terminates DeepCave.
- Nicer handling of Keyboard Interrupt.
- Disable debug mode.
- Save plotly plots in higher resolution upon download.
- Get hovertext per budget in Footprint, Config Cube, Cost over Time, and Pareto Front.

## Bug-Fixes
- Fix missing objective specification in LPI evaluator (#71).
- Don't convert BOHB runs with status 'running' (consistent with SMAC).
- Fix api examples (#68).
- Reset inputs to fix error when subsequently selecting runs with different configspaces, objectives or budgets (#106).
- Fix errors due to changing inputs before runselection (#64).
- For fANOVA, remove constant hyperparameters from configspace (#9).
- When getting budget, objectives etc from multiple runs in Cost over Time and Pareto Front:
    - Instead of taking the first run as comparative value,
    - take the one with the lowest budget, else the index for the budgets could be out of bounds.
- For PCP, show hyperparameters with highest importance closest to the cost, i.e. right (#124).
- Add init files to all test directories.
- Correct LPI importance tests.
- Free port when exiting via the exit button (#52).

## Documentation
- Add How to Contribute section.
- Expand documentation for partial dependence plugin.

## Version-Updates
- Black version from 23.1.0 to 23.3.0
- Mypy from 0.930 to 1.5.1

## Mypy
- Updated args so there are no missing imports
- Updated additional dependencies, so mypy can check all types
- Note: If the installs in requirements change, it has to be adapted in additional dependencies
- Added many type annotations
- In some cases internal refactoring for variables, due to typing

## Pydocstyle and Linter
- Major overhaul of docstrings in various files
- Removed unused imports and variables

## Groups
- Groups now get a default name, if no name was entered by the user.
- Groups get sorted alphabetically to appear more ordered.

## Additional Changes
- Added a "make install examples" in Makefile

# Version 1.1.3

## Bug-Fixes
- Fix seaborn style name (#82).
- Remove potential sources of nondeterminism in evaluators by not setting seeds randomly (#75).
- Exchange SMAC log examples to fix issue with PDP (#54).
- Fix error when requesting more than 10 colors in a plot (36 colors available now).

# Version 1.1.2

## Bug-Fixes
- Fix configspace version.

# Version 1.1.1

## Bug-Fixes
- Put in missing "q" values into configspaces to allow using newer ConfigSpace versions.
- Fix minimal configspace version.
- Fixes in make publish command.

# Version 1.1

## Converters
- SMAC 2.0

## Dependencies
- Remove SMAC dependency by adding required function directly.

# Version 1.0.1

## Bug-Fixes
- SMAC Converter: If config origin is not found then `None` is used.
- Restricting SMAC<1.4.0 as SMAC had some API changes.


# Version 1.0

## General
- Runs are sorted now and only "visible" directories are shown.
- Mechanics to select runs changed.
    - Runs can be selected across different working directories now.
    - Another section to display selected runs.
    - Press on directory name changes the working directory directly. Makes it easier to navigate.
    There's also a button to go to parent directory.
    - Internally, a run has two hashes now
        - `id`: Based on prefix and path/name. This hash is used to select runs now. No need to
        distinguish between run and grouped run anymore.
        - `hash`: Changes based on content.
- If DeepCAVE was not started before, the path of execution is used as working directory.
- Shows the errors now if a run could not be converted.
- Increased stability for static plugins.
- Updated CLI immensively. Added --open, --n_workers and improved --config (both relative
and absolute paths are working now). DeepCAVE starts now using only `deepcave`.
- Improved run cache performance. Each input uses a single file for the output.
- Improved performance overall.
- Improved the API mode drastically. 
- Improved mapping of original and encoded data.
- Configurations are clickable now.
- API changes in `AbstractRun`.
- Jobs in sidebar are clickable and removeable now.

## Documentation
- Updated texts and images.
- Documentation of plugins are shown in the dashboard now (converted from rst to md).
- Added research questions.

## Plugins
- Overview: Redesigned with barplot and heatmap for statuses. Also includes configspace now.
- Added PDP again.
- Merged LPI and fANOVA.
- Added dynamic texts to overview and budget correlation.
- Added help buttons.
- Added configuration footprint.
- Configurations display code now.

## Code related
- Added dash-extensions so that multiple outputs, no outputs and trigger can be used.
- Added global notification.
- Cleaned-up run handler immensely.
- Logs are better readable.
- Improved config loading and add development config.
- Renamed and moved things (Groups, Status, Objectives, ...).
- Objective is a dataclass now.
- Added REFRESH_RATE and SAVE_IMAGES to profile.

## Bugfixes
- Files are no longer shown in run selection.
- Runs are reloaded from disk in worker now.
- Internal runs are updated now if they changed.
- Results from static plugins are saved under the right path now.


# Version 0.3

## General
- Updated tools.
- Cleaned up (removed docker files).

## Installation
- Checks if redis-server is installed. Otherwise, aborts to start the server.
- Added swig to anaconda environment.
- Increased version of numpy and matplotlib.
- Improved requirements. Fixed only crucical things.

## Run
- Added model support.

## Plugins
- Added button to receive raw data.
- Added new plugin: Pareto Front.
- Added new plugin: Budget Correlation.
- Changed `load_outputs` and `load_mlp_outputs` to class method. Moreover, if
`activate_run_selection` is true, only the the run and runs information in `outputs` is passed.
- Cost Over Time displays maximized objectives correctly now.
- Pre-selecting highest budget and hyperparameters (#18).
- Replaced normalized costs with all costs in configurations.

## Examples
- Added MLP recorder example.
- Added MLP logs for better example visualization.

## Bugfixes
- Fixed path bug for SMAC and BOHB.
- Included `start.sh`.
- pytest no longer needed in main code.
- Included logging.yml (before it was loggin.yml).
- Recorder works again.

## Docs
- Added documentation with most basic information.


# Version 0.2

## Runs
- Groups were replaced by `GroupedRun`. Hence, a group can be used as a normal run (`AbstractRun`).
- Groups are only created if the runs are compatible (via function `check_equality`).

## Plugins
- Integrated Parallel Coordinates.
- Configurations Cube and Parallel Coordinates show the real config values and not
the encoded anymore.
- Changed plugin interfaces (check_run_compatibility, check_runs_compatibility,
load_inputs, load_dependency_inputs).
- Enhanced native group selection drastically.
- Added icons to the plugins.
- Removed ICE for now as too many bugs were found.

## Others
- Added typehints.
- Added tests for caches, basic layouts and other utils.
- Added tools (mypy, flake8 and blake)
- Performance increase in `general` as a folder is only transformed to a run if selected.

## Known Bugs
- fANOVA does not work if the configspace includes constants.


# Version 0.1.1

## Plugins

- ICE/PDP: Added uncertainties
- fANOVA: Works with no budgets now

## Others

- Installable via pip


# Version 0.1

## Features

- Multi-Objective is supported with flexible objective objects.
- Groups: If plugin supports groups, multiple runs can be used.
- Automatically find converter: DeepCAVE scans the given directory and selects a suitable converter (if available).
- Recorder: Record your runs and save them directly as DeepCAVE files.
- Matplotlib mode: Alternatively, plugins can output Matplotlib rendered plots.

## Plugins

- StaticPlugin: Uses queue to process the result.
- DynamicPlugin: Input changes are directly calculated.

- Overview: Shows meta and statistics.
- Configurations: Shows configuration space and best found configuration.
- Cost over Time: Shows how the cost changes over time.
- Configuration Cube: Shows configurations as points in a cube.
- Individual Conditional Expectation / Partial Dependency Plots: Shows how individual instances behave.
- fANOVA: Shows Hyperparameter Importance.

## Converters

- DeepCAVE
- SMAC
- BOHB (Beta)
