from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as image
import matplotlib.font_manager as fm

from canalyst_candas.settings import BRAND_CONFIG_DEFAULTS

plt.style.use("fivethirtyeight")


class Chart:
    """
    Charting class to create consistent and branded charts

    Attributes
    ----------
    x_value : pd.Series
        the data for the x-axis
    y_values : pd.DataFrame
        the data for the y-axis
    labels : list
        a list of strings to specify labels of y_value columns respectively (Default: None)
    axis_labels : list
        a list of tuples of strings to specify x,y axis labels (Default: None)
    title : str
        the title of the chart (Default: None)
    plot_styles : list
        list of chart styles for y-value columns (can specify only 1 style to apply to all columns) (Default: ["line"])
    use_subplots : bool
        bool to indicate if subplots should be used (i.e chart columns on the same graph or not) (Default: False)
    subplot_scale_width : list
        list of ints to specify width scaling for subplots (length of list must match # of subplots) (Default: None)
    subplot_scale_height : list
        list of ints to specify height scaling for subplots (length of list must match # of subplots) (Default: None)
    figure_size : tuple
        tuple to define figure size (Default: (12,9))
    vertical_line : str or pd.Series
        str or DataFrame series to define where to place a vertical line on the graph on the x-axis (Default: None)
    marker : str
        str to define marker type (Default: "o")
    markersize : int
        the size of markers (Default: 8)
    markevery : int or float
        int argument will define plotting of every nth marker from the first data point.  (Default: None)
    plot_config : dict
        extra arguments to be passed into the .plot() command. Arguments must be valid for the plot style being used. (Default: None)
    xtick_rotation : int
        specifies the rotation of x-axis tick labels (Default: 90)
    display_charts_horizontal : bool
        specifies if subplots should be displayed horizontally instead of vertically (Default: False)
    subplot_adjustment : list
        list in the form of [left, bottom, right, top] to tune subplot layout. (Default: [0.125, 0.2, 0.9, 0.9])

    Methods
    -------
    show()
        Displays the graph generated
    build_chart(force)
        Builds a matplotlib chart. if force is set to True, chart will be built from scratch regardless if cache is present.
    """

    def __init__(
        self,
        x_value,
        y_values,
        labels=None,
        axis_labels=None,
        title="",
        plot_styles=[
            "line"
        ],  # If only 1 style defined, it will be applied to all columns
        use_subplots=False,
        subplot_scale_width=None,
        subplot_scale_height=None,
        figure_size=(12, 9),
        vertical_line=None,
        marker="o",
        markersize=8,
        markevery=None,
        plot_config={},
        xtick_rotation=90,
        display_charts_horizontal=False,
        subplot_adjustment=[0.125, 0.2, 0.9, 0.9],
        brand_config=None,
        include_logo_watermark=True,
    ):

        self.x_value = x_value
        self.y_values = y_values
        self.labels = labels
        self.axis_labels = axis_labels
        self.title = title
        self.use_subplots = use_subplots
        self.plot_styles = plot_styles
        self.subplot_scale_width = subplot_scale_width
        self.subplot_scale_height = subplot_scale_height
        self.figure_size = figure_size
        self.vertical_line = vertical_line
        self.markevery = markevery
        self.marker = marker
        self.markersize = markersize
        self.plt_plot_style = "fivethirtyeight"
        self.plot_config = plot_config
        self.xtick_rotation = xtick_rotation
        self.display_charts_horizontal = display_charts_horizontal
        self.subplot_adjustment = subplot_adjustment
        self.include_logo_watermark = include_logo_watermark

        self.brand_config = brand_config or BRAND_CONFIG_DEFAULTS

        self._validate_brand_config()
        self._validate_args()

        self.main_fpath = Path(self.brand_config["title_font_path"])
        self.secondary_fpath = Path(self.brand_config["body_font_path"])

        plt.rc("figure", figsize=self.figure_size)
        self.fig, self.axs = self.build_chart()

    def _validate_brand_config(self):
        try:
            self.brand_config["title_font_path"]
        except KeyError:
            raise KeyError(
                "Title font path not found in chart brand configuration. Please reset or update chart brand configs"
            )

        try:
            self.brand_config["chart_bg_color"]
        except KeyError:
            raise KeyError(
                "Chart background color not found in chart brand configuration. Please reset or update chart brand configs"
            )

        try:
            self.brand_config["body_font_path"]
        except KeyError:
            raise KeyError(
                "Body font path not found in chart brand configuration. Please reset or update chart brand configs"
            )

        try:
            self.brand_config["logo_path"]
        except KeyError:
            raise KeyError(
                "Logo path not found in configuration. Please reset or update configs."
            )

        try:
            self.brand_config["figure_bg_color"]
        except KeyError:
            raise KeyError(
                "Figure background color not found in configuration. Please reset or update configs."
            )

        try:
            self.brand_config["font_color"]
        except KeyError:
            raise KeyError(
                "Font color not found in configuration. Please reset or update configs."
            )

        try:
            self.brand_config["vertical_line_color"]
        except KeyError:
            raise KeyError(
                "Vertical line color not found in configuration. Please reset or update configs."
            )

        try:
            self.brand_config["chart_plot_colors"]
        except KeyError:
            raise KeyError(
                "First chart plot color not found in configuration. Please reset or update configs."
            )

        try:
            self.brand_config["ax_spine_color"]
        except KeyError:
            raise KeyError(
                "Ax spine color not found in configuration. Please reset or update configs."
            )

    def _validate_args(self):
        """
        Validate arguments used in initialization
        """
        if not type(self.plot_styles) == list:
            print(
                "Plot styles must be specified in the form of a list. Please re-instantiate the "
                'object with plot style in the form of "["plot_style_here"]". Available options '
                'are "line", "bar", "barh", "scatter".'
            )
        if not type(self.labels) == list:
            print(
                "Labels must be specified in the form of a list. Please re-instantiate the object"
                'with labels in the form of "["Label", "Label"]" depending on how many columns'
                "are being passed in."
            )

        if isinstance(self.y_values, pd.DataFrame):
            if type(self.labels) == list and not len(self.labels) == len(
                self.y_values.columns
            ):
                print("Warning: Labels have not been appointed for all columns.")

        if isinstance(self.y_values, pd.Series):
            if type(self.plot_styles) == list and len(self.plot_styles) > 1:
                print(
                    "Only 1 plot style needed for the specified y data. Only the first value "
                    "in the list will be used."
                )
        elif isinstance(self.y_values, pd.DataFrame):
            columns = len(self.y_values.columns)
            # If user has specified only 1 plot style for multiple columns, extend it to all columns
            if type(self.plot_styles) == list and len(self.plot_styles) == 1:
                self.plot_styles = self.plot_styles * columns
            elif (
                not type(self.plot_styles) == list
                or not len(self.plot_styles) == columns
            ):
                print(
                    "Please explicitly specify the plot style of all Y value columns in a list "
                    "or 1 style to be applied for all."
                )

    def _add_watermark(self, ax):
        """
        Adds a Canalyst Logo to the top left corner of the graph
        """

        logo_path = self.brand_config["logo_path"]

        logo = image.imread(logo_path)
        starting_x, starting_y, width, height = ax.get_position().bounds

        # Units seem to be in inches:
        # https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/figure_size_units.html#:~:text=The%20native%20figure%20size%20unit,units%20like%20centimeters%20or%20pixels.
        logo_width = 0.06
        logo_height = 0.06

        # Change the numbers in this array to position your image [left, bottom, width, height])
        ax = plt.axes(
            [
                starting_x + 0.01,
                starting_y + height - logo_height,
                logo_width,
                logo_height,
            ],
            frameon=True,
        )
        ax.imshow(logo)
        ax.axis("off")

    def setup_basic_chart_options(self, fig):
        """
        Sets up figure with basic chart options
        """
        plt.style.use(self.plt_plot_style)

        fig_bg_color = self.brand_config["figure_bg_color"]

        fig.patch.set_facecolor(fig_bg_color)
        # Adjust subplot chart display positions
        left, bottom, right, top = self.subplot_adjustment
        plt.subplots_adjust(left, bottom, right, top)

        font_color = self.brand_config["font_color"]

        mpl.rcParams["text.color"] = font_color
        mpl.rcParams["axes.labelcolor"] = font_color
        mpl.rcParams["xtick.color"] = font_color
        mpl.rcParams["ytick.color"] = font_color

    def build_chart(self, force=False):
        """
        Builds a matplotlib chart. Returns figure and ax(s).
        """

        # Avoid rebuilding if not required
        try:
            if self.fig and self.axs and not force:
                print(
                    f"Skipping re-build as the chart is already built. "
                    f"Please provide a `force=True` argument to force the rebuilding of the chart"
                )
                return self.fig, self.axs
        except AttributeError:
            pass

        if isinstance(self.y_values, pd.Series):
            fig, ax = self._get_chart_without_subplots(is_series=True)
        elif isinstance(self.y_values, pd.DataFrame):
            if not self.use_subplots:
                fig, ax = self._get_chart_without_subplots(is_series=False)
            else:
                # Use subplots
                fig, ax = self._get_chart_with_subplots()

        # If a vertical line is specified, and there are subplots: apply it on all subplots
        # If not, apply it to just the one graph.
        vertical_line_color = self.brand_config["vertical_line_color"]

        if self.vertical_line:
            if type(ax) == np.ndarray:
                for index, item in enumerate(ax):
                    ax[index].axvline(
                        x=self.vertical_line,
                        color=vertical_line_color,
                    )
            else:
                ax.axvline(
                    x=self.vertical_line,
                    color=vertical_line_color,
                    linewidth=2,
                )

        return fig, ax

    def _get_axis_label(self, index):
        """
        Return axis label at position specified.

        Returns None if class not initialized with axis_labels.
        """
        if self.axis_labels is not None:
            try:
                return self.axis_labels[index]
            except IndexError:
                pass

        return None

    def _get_label(self, index):
        """
        Return label at position specified.

        Returns None if class not initialized with labels.
        """
        if self.labels is not None:
            try:
                return self.labels[index]
            except IndexError:
                pass
        return None

    def _get_chart_without_subplots(self, is_series=True):
        """
        Returns figure and ax for charts without subplots for series or DataFrames
        """
        fig, ax = plt.subplots()
        first_element = 0

        font_color = self.brand_config["font_color"]

        # Get and set custom font
        fig.suptitle(
            self.title,
            font=self.main_fpath,
            size="x-large",
            weight="bold",
            y=0.96,
            color=font_color,
        )

        self.setup_basic_chart_options(fig)

        label = self._get_label(first_element)
        axis_label = self._get_axis_label(first_element)

        first_chart_plot_color = self.brand_config["chart_plot_colors"][0]

        # If series is specified, we only have 1 y-value to graph
        if is_series:
            self._set_graph_ax(
                ax,
                self.plot_styles[0],
                self.x_value,
                self.y_values,
                label,
                axis_label,
                first_chart_plot_color,
                self.marker,
                self.markersize,
                self.markevery,
            )
        else:
            # Want to graph multiple y-values on the same graph
            color_count = len(self.brand_config["chart_plot_colors"])
            for index, column in enumerate(self.y_values):
                label = self._get_label(index)
                self._set_graph_ax(
                    ax,
                    self.plot_styles[index],
                    self.x_value,
                    self.y_values[column],
                    label,
                    axis_label,
                    self.brand_config["chart_plot_colors"][
                        index % color_count
                    ],  # Use index modded by length of color_options to select unique colors, up to color_count, then colors repeat.
                    self.marker,
                    self.markersize,
                    self.markevery,
                )

        self._format_graph(ax, fig)
        return fig, ax

    def _get_scale_arguments(self):
        """
        Return width, height scale arguments, if any
        """
        scale_args = {}

        if self.subplot_scale_width and self.subplot_scale_height:
            scale_args = {
                "gridspec_kw": {
                    "width_ratios": self.subplot_scale_width,
                    "height_ratios": self.subplot_scale_height,
                }
            }
        elif self.subplot_scale_width:
            scale_args = {"gridspec_kw": {"width_ratios": self.subplot_scale_width}}
        elif self.subplot_scale_height:
            scale_args = {"gridspec_kw": {"height_ratios": self.subplot_scale_height}}

        return scale_args

    def _get_chart_with_subplots(self):
        """
        Returns figure and ax for charts with subplots for DataFrames
        """
        color_count = len(self.brand_config["chart_plot_colors"])
        columns = len(self.y_values.columns)
        scale_args = self._get_scale_arguments()

        # If specified, display subplots horizontally
        if columns % 2 == 0 and self.display_charts_horizontal:
            row = int(columns / 2)
            row_column = {"nrows": row, "ncols": columns}
        else:
            row_column = {"nrows": columns}

        fig, axs = plt.subplots(
            **row_column,
            **scale_args,
            sharex=True,
        )

        font_color = self.brand_config["font_color"]

        # Set custom Font and figure title
        fig.suptitle(
            self.title,
            font=self.main_fpath,
            size="x-large",
            weight="bold",
            y=0.96,
            color=font_color,
        )

        self.setup_basic_chart_options(fig)

        # Plot each column from the Y-column dataset provided
        for index, column in enumerate(self.y_values):
            label = self._get_label(index)
            axis_label = self._get_axis_label(index)

            self._set_graph_ax(
                axs[index],
                self.plot_styles[index],
                self.x_value,
                self.y_values[column],
                label,
                axis_label,
                self.brand_config["chart_plot_colors"][index % color_count],
                self.marker,
                self.markersize,
                self.markevery,
            )
            self._format_graph(axs[index], fig)

        return fig, axs

    def _set_graph_ax(
        self,
        ax,
        plot_style,
        x_value,
        y_value,
        label,
        axis_labels,
        color,
        marker,
        markersize,
        markevery,
    ):
        """
        Builds a matplotlib chart on the ax provided based on configurations specified.
        """
        font_color = self.brand_config["font_color"]

        # Set up axis labels
        if axis_labels is not None:
            x_label, y_label = axis_labels
            ax.set_xlabel(
                x_label,
                font=self.main_fpath,
                labelpad=15,
                color=font_color,
            )
            ax.set_ylabel(
                y_label,
                font=self.main_fpath,
                labelpad=15,
                color=font_color,
            )

        # Plot the type of bar specified
        if plot_style == "bar":
            ax.bar(
                x_value,
                y_value,
                label=label,
                color=color,
                width=0.4,
                zorder=-1,
                **self.plot_config,
            )
        elif plot_style == "scatter":
            ax.scatter(
                x_value,
                y_value,
                label=label,
                color=color,
                zorder=3,
            )
        elif plot_style == "barh":
            ax.barh(
                x_value,
                y_value,
                label=label,
                color=color,
                zorder=-2,
                **self.plot_config,
            )
        else:  # line graph or catch all
            ax.plot(
                x_value,
                y_value,
                label=label,
                color=color,
                marker=marker,
                markersize=markersize,
                markevery=markevery,
                linewidth=2,
                zorder=1,
                **self.plot_config,
            )

    def _format_graph(self, ax, fig):
        """
        Formats ax's
        """
        ax_spine_color = self.brand_config["ax_spine_color"]

        chart_bg_color = self.brand_config["chart_bg_color"]

        # General grid and chart options
        ax.grid(linewidth=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(ax_spine_color)
        ax.spines["bottom"].set_color(ax_spine_color)
        ax.set_facecolor(chart_bg_color)

        # Setup Legend
        columns = 1
        if isinstance(self.y_values, pd.DataFrame):
            columns = len(self.y_values.columns)

        # Set custom Font
        font_name = fm.FontProperties(fname=self.secondary_fpath)

        fig.legend(
            loc="lower left",
            bbox_to_anchor=(0.1, 0),
            fancybox=True,
            shadow=True,
            ncol=columns,
            frameon=False,
            prop=font_name,
        )

        font_color = self.brand_config["font_color"]

        # Set up x,y axis tick labels
        config_ticks = {
            "size": 0,
            "labelcolor": font_color,
            "labelsize": 12,
            "pad": 10,
        }
        ax.tick_params(axis="both", **config_ticks)

        if self.include_logo_watermark:
            # Add Canalyst watermark
            self._add_watermark(ax)

        for tick in ax.get_xticklabels():
            tick.set_font(self.secondary_fpath)
        for tick in ax.get_yticklabels():
            tick.set_font(self.secondary_fpath)

        # Set the rotation for tick labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(self.xtick_rotation)

    def show(self):
        """
        Displays Chart
        """
        plt.show()
