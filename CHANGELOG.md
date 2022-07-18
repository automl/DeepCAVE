# Version 1.0.1

## Bug-Fixes
- SMAC Converter: If config origin is not found then `None` is used.


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
