#  noqa: D400
"""
# StyledPlot

This module provides utilities to customize functions from the matplotlib.
Plots can be created and different parameters of the plots can be defined.

## Classes
    - StyledPlot: Overwrites default settings from matplotlib.pyplot.

## Constants
    FIG_WIDTH: float
    FIG_HEIGHT: float
"""

from typing import Any, Dict, Optional, Union

import base64
import io
from distutils.spawn import find_executable
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from deepcave.utils.logs import get_logger

matplotlib.use("Agg")


# IEEETrans double column standard
FIG_WIDTH = 252.0 / 72.27  # 1pt is 1/72.27 inches
FIG_HEIGHT = FIG_WIDTH / 1.618  # golden ratio


logger = get_logger(__name__)


class StyledPlot:
    """
    Overwrites default settings from matplotlib.pyplot.

    If a function is not overwritten, the default function will be used.

    Properties
    ----------
    plt : Module("matplotlib.pyplot")
        The matplotlib plot.
        The style used is seaborn.
    """

    def __init__(self) -> None:
        plt.style.use("seaborn-v0_8")

        # Set MatPlotLib defaults
        if find_executable("latex"):
            plt.rcParams.update(
                {
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern"],
                }
            )
        else:
            logger.warn("LaTeX not found. Using default font.")

        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        plt.rc("axes", labelsize=12)
        plt.rc("axes", titlesize=12)
        plt.rc("legend", fontsize=8)

        self.plt = plt

    def figure(self, cols: int = 1, rows: int = 1, dpi: int = 200) -> plt.Figure:
        """
        Create a new figure using the input values.

        Parameters
        ----------
        cols : int, optional
            The number of the columns.
            Default is 1.
        rows : int, optional
            The number of the rows.
            Default is 1.
        dpi : int, optional
            The dots per inches (dpi).
            Default is 200.

        Returns
        -------
        plt.Figure
            The figure created with the input information.
        """
        # Clean all
        self.plt.cla()
        self.plt.clf()

        f = self.plt.figure(figsize=(FIG_WIDTH * cols, FIG_HEIGHT * rows), dpi=dpi)
        f.tight_layout()

        return f

    def save_figure(self, filename: Union[str, Path]) -> None:
        """
        Save the figure/plot at the given filename.

        Parameters
        ----------
        filename : Union[str, Path]
            The name of the file the plot will be saved at.
        """
        self.plt.savefig(filename, dpi=400, bbox_inches="tight")
        self.plt.close()

    def render(self) -> str:
        """
        Render the Styled Plot for displaying.

        Returns
        -------
        str
            The rendered plot.
        """
        # Create a virtual file which matplotlib can use to save the figure
        buffer = io.BytesIO()
        self.plt.savefig(buffer, dpi=400, bbox_inches="tight")
        buffer.seek(0)

        # Display any kind of image taken from
        # https://github.com/plotly/dash/issues/71
        encoded_image = base64.b64encode(buffer.read())
        return "data:image/png;base64,{}".format(encoded_image.decode())

    def xlim(self, xmin: Union[float, int], xmax: Union[float, int]) -> None:
        """
        Set the x-axis limits with a margin of a matplotlib plot.

        Parameters
        ----------
        xmin : Union[float, int]
            The lower x-axis limit.
        xmax : Union[float, int]
            The upper x-axis limit.
        """
        xmin_with_margin = xmin - 0.05 * (xmax - xmin)
        xmax_with_margin = xmax + 0.05 * (xmax - xmin)
        self.plt.xlim(xmin_with_margin, xmax_with_margin)

    def ylim(self, ymin: Union[float, int], ymax: Union[float, int], margin: bool = True) -> None:
        """
        Set the y-axis limit of a matplotlib plot.

        Parameters
        ----------
        ymin : Union[float, int]
            The lower y-axis limit.
        ymax : Union[float, int]
            The upper y-axis limit.
        margin : bool, optional
            Determines whether a margin should be added to the limits.
            Default is True.
        """
        if margin:
            ymin_with_margin = ymin - 0.05 * (ymax - ymin)
            ymax_with_margin = ymax + 0.05 * (ymax - ymin)
            self.plt.ylim(ymin_with_margin, ymax_with_margin)
        else:
            self.plt.ylim(ymin, ymax)

    # def grid(self):
    #    pass
    #    # self.plt.grid(b=True, color='black', linestyle='--', linewidth=0.5, axis='y', zorder=0,
    #    # alpha=0.5)

    def boxplot(
        self, values: np.ndarray, positions: ArrayLike, color: str, widths: float = 0.5
    ) -> None:
        """
        Create a boxplot on a matplotlib plot.

        Parameters
        ----------
        values : np.ndarray
            Values to create the boxplot.
        positions : ArrayLike
            The position of the boxplot.
        color : str
            The color of the boxes as well as other elements in the plot.
        widths : float, optional
            The width of the boxes.
            Default is 0.5.
        """
        bp = self.plt.boxplot(values, positions=positions, patch_artist=True, widths=widths)

        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set(linewidth=0.3)

        for whisker in bp["whiskers"]:
            whisker.set(color=color, linewidth=0.5)

        for cap in bp["caps"]:
            cap.set(color=color, linewidth=0.5)

        for median in bp["medians"]:
            median.set(color="black", linewidth=0.5)

        for flier in bp["fliers"]:
            flier.set(
                marker="o",
                markersize=3,
                markerfacecolor=color,
                linestyle="none",
                markeredgecolor="none",
                color=color,
                alpha=0.5,
            )

    def legend(
        self,
        cols: int = 1,
        loc: Optional[str] = None,
        title: Optional[str] = None,
        outside: bool = False,
    ) -> None:
        """
        Customize and add a legend to a matplot plot.

        Customize the placement and appearance of the legend.

        Parameters
        ----------
        cols : int, optional
            The number of the columns.
            Default is 1
        loc : Optional[str], optional
            The location of the legend.
            Default is None.
        title : Optional[str], optional
            The title for the legend.
            Default is None.
        outside : bool, optional
            Determines if a legend is placed outside of plot area.
            Default is False.
        """
        kwargs: Dict[str, Any] = {
            "ncol": cols,
            "columnspacing": 0.8,
            "labelspacing": 0,
            "loc": loc,
            "fancybox": False,
            "framealpha": 0.8,
            "frameon": True,
            "borderaxespad": 0.4,
            "facecolor": "white",
            "title": title,
        }

        if loc is not None:
            kwargs["loc"] = loc

        if outside:
            kwargs.update({"loc": "upper left", "bbox_to_anchor": (1, 1)})

        legend = self.plt.legend(**kwargs)
        legend.set_zorder(500)

        if outside:
            legend.get_frame().set_linewidth(0.0)
            legend.get_frame().set_edgecolor("white")
        else:
            legend.get_frame().set_linewidth(0.5)
            legend.get_frame().set_edgecolor("gray")

    def get_color(self, id: int) -> str:
        """
        Get the color from color palette based on the given id.

        Parameters
        ----------
        id : int
            id for retrieving a specific color.

        Returns
        -------
        str
            The color from the color palette.
        """
        import seaborn as sns

        pal = sns.color_palette()
        hex_codes = pal.as_hex()
        return hex_codes[id % len(hex_codes)]

    def __getattr__(self, name: str) -> Any:
        """Make sure self.plt is accessed directly."""
        try:
            return self.__getattribute__(name)
        except AttributeError:
            return self.plt.__getattribute__(name)


plot = StyledPlot()
