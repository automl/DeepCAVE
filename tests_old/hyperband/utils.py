import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
plt.style.use('seaborn-whitegrid')


def plot_grey_box_optimization(configs_list, min_budget_per_model):
    if len(configs_list) == 1:
        n_rows, n_cols = 1, 1
        filename = 'successive_halving_results.pdf'
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5, 5), sharex='col', sharey='row')
        axs = [axs]
    else:
        n_hyperband_iter = len(configs_list)
        n_cols = ((n_hyperband_iter - np.mod(n_hyperband_iter, 3)) / 3 + np.mod(n_hyperband_iter, 3)).astype(np.int)
        n_rows = 3
        filename = 'hyperband_results.pdf'
        fig, axs = plt.subplots(3, n_cols, figsize=(n_cols * n_rows, n_rows * 2), sharex='col', sharey='row')
        axs = axs.reshape(-1)

    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for idx, (configs_dict, ax) in enumerate(zip(configs_list, axs)):
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid(True, which="both", ls="-", alpha=0.3)
        all_budgets = []
        for config_id, config_dict in configs_dict.items():
            budgets = np.array(list(config_dict['f_evals'].keys()))
            val_errors = np.array(list(config_dict['f_evals'].values())).T
            ax.scatter(budgets, val_errors, s=6)
            ax.plot(budgets, val_errors)
            all_budgets.extend(budgets)

        # Use the same x-axis limits for all subplots for easier comparison.
        ax.set_xlim(min_budget_per_model, 110)

        for budget in np.unique(all_budgets):
            ax.axvline(budget, c='black', lw=0.5)
        if idx == 0:
            ax.set_xlabel('Budget (Epochs)')
            ax.set_ylabel('Validation Error')
    plt.tight_layout()
    plt.savefig(filename)
    return fig
