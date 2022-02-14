# v0.2.1

## Plugins
- Added button to receive raw data

## Bugfixes
- Fixed path bug for SMAC and BOHB


# v0.2

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


# 0.1.1

## Plugins

- ICE/PDP: Added uncertainties
- fANOVA: Works with no budgets now

## Others

- Installable via pip


# v0.1

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
