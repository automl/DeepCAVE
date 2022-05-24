# Overview
* When using a group object with DeepCave runs, the *Optimizer* advertized in the "Quick Information" box is just "group".
* The __(Details)__ in the Quick Information box could be changed to "See configuration" to know that you're jumping to the best configuration.
* The **Quick Information** part could add a line on how many trials were performed, I can't seem to see this number anywhere, it would help distinguish between _trials_ and _configurations_.
* The **Meta** header should probably have a little Q mark or be renamed to **Meta Information**
    * Change `Overview.md` if changing this
    * Need more information about what's actually possible to see here

# Cost over time
* Does the budget help message about time objective being skewed with respect to combined budget still matter in the importances?

# Configuration Cube
* The budget drop down has an option _Combined_, not intuitevly sure what this means, could add a little help to it.
* The yellow dots in configuration cube are increidbly hard to see with a large screen in 2D. Could maybe up the size of them, add borders to the dots or remove any alpha if there is.
* Would you think it makes sense to add slight lines from the point to the axis in 3d to see where each point lines up?

# Importances
* Does the budget help message about time objective being skewed with respect to combined budget still matter in the importances?

# Configuration Footprint
* We should change *Configuration* to **Evaluated Configuration** and *Random (unevaluated) Configuration* to **Unevaluated Configuration**.

# Budget Correlation
* We should have vertical lines at each budget level to make things clearer where the budgets are.
* Perhaps a line plot is not the best for this and a bar chart might be more useful?

# plugins Index.rst
* Image for blocks needs to be updated.
* TOC tree for plugins.index.toctree was corrupted when writing markdown help

# Getting Started
* Needs new screenshot
* Add some a note about some basic initial runs the user can use to explore

# Custom config
* Update how to add your own plugins/config in plugins.index.rst
* Getting started, I see you can do `$ deepcave --start --config=X` but that this `config` needs to be located in `deepcave/configs/<X>.py`. Could this just be made into a path?

# Other
* Breif look over API, some functions such as `Run` which is required for creating a Convertor needs more doc, has TODO notes in it.
* Add walkthrough section explaining an analysis report of a basic run.
* Examples need some text
* Need some information on how to write your own plugin and add it.
* Convertor has the same issue as creating a plugin in that it needs to be added to the `config.py`. Could do with a simple example of creating one and adding it to DeepCave for use.

# Glossary terms to add
* Trial
* Configuration
* Pareto Front
* ...

# FAQ needs more and links

