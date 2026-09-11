import logging
from typing import Deque
from collections import deque, defaultdict

import pandas as pd
import plotly_express as px
import wandb

logger = logging.getLogger(__name__)


class Point:
    def __init__(self, x, y, args=None):
        self.x = x
        self.y = y
        self.args = args if args is not None else {}

    def __eq__(self, other: 'Point'):
        return (
            self.x == other.x and \
            self.y == other.y and \
            self.args == other.args
        )


class DataSeries:

    def __init__(
        self,
        x_name: str,
        y_name: str,
        args: dict={},
    ) -> None:
        # We store every data point in an instance of Point
        self.data_series: Deque[Point] = deque()
        # column names of x, y datatpoints for data collection
        self.x_name = x_name
        self.y_name = y_name
        self.args = args

        # most recently collected y datapoint for incremental updates
        # to aid incremental updates to y datapoints
        self._last_data_y = 0

    def consolidate(
        self,
    ):
        res = defaultdict(list)
        for point in self.data_series:
            res[point.x].append(point.y)
        # NOTE: When consolidating, we remove the args to Point
        # There is no obvious way to combine multiple points with possibly
        # different args into a single point.
        self.data_series = [Point(x, sum(y) / len(y)) for x, y in res.items()]

        # sort by x
        self.data_series = sorted(self.data_series, key=lambda point: point.x)
        self._last_data_y = self.data_series[-1].y if len(self.data_series) else 0

    def merge(self, other: "DataSeries"):
        if len(other) == 0:
            return

        assert self.x_name == other.x_name
        assert self.y_name == other.y_name

        self.data_series.extend(other.data_series)

        # sort by y
        self.data_series = sorted(self.data_series, key=lambda point: point.x)
        self._last_data_y = self.data_series[-1].y

    # This function assumes that x's are unique
    # in their own dataseries respectively.
    def elementwise_merge(self, other: "DataSeries"):
        if len(other) == 0:
            return

        assert self.x_name == other.x_name
        assert self.y_name == other.y_name
        self.data_series.extend(other.data_series)

        res = defaultdict(list)
        for point in self.data_series:
            res[point.x].append(point.y)
        self.data_series = [Point(x, sum(y) / len(y)) for x, y in res.items()]

        # sort by x
        self.data_series = sorted(self.data_series, key=lambda point: point.x)
        self._last_data_y = self.data_series[-1].y

    @property
    def min_x(self):
        if len(self.data_series) == 0:
            return 0

        return self.data_series[0].x

    def __len__(self):
        return len(self.data_series)

    @property
    def sum(self) -> float:
        return sum([point.y for point in self.data_series])

    @property
    def metric_name(self) -> str:
        return self.y_name

    # add a new x, y datapoint
    def put(self, data_x: float, data_y: float) -> None:
        self._last_data_y = data_y
        self.data_series.append(Point(data_x, data_y, self.args))

    # For compatibility with CDFSketch
    def put_pair(self, data_x: float, data_y: float) -> None:
        self.put(data_x, data_y)

    # get most recently collected y datapoint
    def _peek_y(self):
        return self._last_data_y

    # convert list of x, y datapoints to a pandas dataframe
    def to_df(self):
        columns = [self.x_name, self.y_name, *list(self.args.keys())]
        df = pd.DataFrame([{**{self.x_name: p.x, self.y_name: p.y}, **p.args} for p in self.data_series], columns=columns)
        return df

    # add a new x, y datapoint as an incremental (delta) update to
    # recently collected y datapoint
    def put_delta(self, data_x: float, data_y_delta: float) -> None:
        last_data_y = self._peek_y()
        data_y = last_data_y + data_y_delta
        self.put(data_x, data_y)

    def print_series_stats(
        self, df: pd.DataFrame, plot_name: str, y_name: str = None
    ) -> None:

        if len(self.data_series) == 0:
            return

        if y_name is None:
            y_name = self.y_name
        try:
            logger.info(
                f"{plot_name}: {y_name} stats:"
                f" min: {df[y_name].min()},"
                f" max: {df[y_name].max()},"
                f" mean: {df[y_name].mean()},"
            )
        except:
            pass
        if wandb.run:
            wandb.log(
                {
                    f"{plot_name}_min": df[y_name].min(),
                    f"{plot_name}_max": df[y_name].max(),
                    f"{plot_name}_mean": df[y_name].mean(),
                },
                step=0,
            )

    def print_distribution_stats(
        self, df: pd.DataFrame, plot_name: str, y_name: str = None
    ) -> None:

        if len(self.data_series) == 0:
            return

        if y_name is None:
            y_name = self.y_name

        try:
            logger.info(
                f"{plot_name}: {y_name} stats:"
                f" min: {df[y_name].min()},"
                f" max: {df[y_name].max()},"
                f" mean: {df[y_name].mean()},"
                f" median: {df[y_name].median()},"
                f" 95th percentile: {df[y_name].quantile(0.95)},"
                f" 99th percentile: {df[y_name].quantile(0.99)}"
                f" 99.9th percentile: {df[y_name].quantile(0.999)}"
            )
        except:
            pass

        if wandb.run:
            wandb.log(
                {
                    f"{plot_name}_min": df[y_name].min(),
                    f"{plot_name}_max": df[y_name].max(),
                    f"{plot_name}_mean": df[y_name].mean(),
                    f"{plot_name}_median": df[y_name].median(),
                    f"{plot_name}_95th_percentile": df[y_name].quantile(0.95),
                    f"{plot_name}_99th_percentile": df[y_name].quantile(0.99),
                    f"{plot_name}_99.9th_percentile": df[y_name].quantile(0.999),
                },
                step=0,
            )

    def _save_df(self, df: pd.DataFrame, path: str, plot_name: str) -> None:
        df.to_csv(f"{path}/{plot_name}.csv", index=False)

    def save_df(self, path: str, plot_name: str) -> None:
        df = self.to_df()
        self._save_df(df, path, plot_name)

    def plot_step(
        self,
        path: str,
        plot_name: str,
        y_axis_label: str = None,
        start_time: float = 0,
        y_cumsum: bool = True,
    ) -> None:

        if len(self.data_series) == 0:
            return

        if y_axis_label is None:
            y_axis_label = self.y_name

        df = self.to_df()

        df[self.x_name] -= start_time

        if y_cumsum:
            df[self.y_name] = df[self.y_name].cumsum()

        # self.print_series_stats(df, plot_name)

        # change marker color to red
        fig = px.line(
            df, x=self.x_name, y=self.y_name, markers=True, labels={"x": y_axis_label}
        )
        fig.update_traces(marker=dict(color="red", size=2))

        if wandb.run:
            wandb_df = df.copy()
            # rename the self.y_name column to y_axis_label
            wandb_df = wandb_df.rename(columns={self.y_name: y_axis_label})

            wandb.log(
                {
                    f"{plot_name}_step": wandb.plot.line(
                        wandb.Table(dataframe=wandb_df),
                        self.x_name,
                        y_axis_label,
                        title=plot_name,
                    )
                },
                step=0,
            )

        fig.write_image(f"{path}/{plot_name}.png")
        self._save_df(df, path, plot_name)

    def plot_cdf(self, path: str, plot_name: str, y_axis_label: str = None) -> None:

        if len(self.data_series) == 0:
            return

        if y_axis_label is None:
            y_axis_label = self.y_name

        df = self.to_df()

        # self.print_distribution_stats(df, plot_name)

        df["cdf"] = df[self.y_name].rank(method="first", pct=True)
        # sort by cdf
        df = df.sort_values(by=["cdf"])

        # Bound rendering cost while retaining the full CDF for CSV export.
        # Sample evenly across ranks, including both endpoints.
        max_plot_points = 2000
        plot_df = df
        if len(df) > max_plot_points:
            indices = [
                i * (len(df) - 1) // (max_plot_points - 1)
                for i in range(max_plot_points)
            ]
            plot_df = df.iloc[indices]

        fig = px.line(
            plot_df, x=self.y_name, y="cdf", markers=False, labels={"x": y_axis_label}
        )

        if wandb.run:
            wandb_df = plot_df.copy()
            # rename the self.y_name column to y_axis_label
            wandb_df = wandb_df.rename(columns={self.y_name: y_axis_label})

            wandb.log(
                {
                    f"{plot_name}_cdf": wandb.plot.line(
                        wandb.Table(dataframe=wandb_df),
                        "cdf",
                        y_axis_label,
                        title=plot_name,
                    )
                },
                step=0,
            )

        fig.write_image(f"{path}/{plot_name}.png")
        self._save_df(df, path, plot_name)

    def plot_histogram(self, path: str, plot_name: str) -> None:
        if len(self.data_series) == 0:
            return

        df = self.to_df()

        # self.print_distribution_stats(df, plot_name)

        fig = px.histogram(df, x=self.y_name, nbins=25)

        # wandb histogram is highly inaccurate so we need to generate the histogram
        # ourselves and then use wandb bar chart

        histogram_df = df[self.y_name].value_counts(bins=25, sort=False).sort_index()
        histogram_df = histogram_df.reset_index()
        histogram_df.columns = ["Bins", "count"]
        histogram_df["Bins"] = histogram_df["Bins"].apply(lambda x: x.mid)
        histogram_df = histogram_df.sort_values(by=["Bins"])
        # convert to percentage
        histogram_df["Percentage"] = histogram_df["count"] * 100 / len(df)
        # drop bins with less than 0.1% of the total count
        histogram_df = histogram_df[histogram_df["Percentage"] > 0.1]

        if wandb.run:
            wandb.log(
                {
                    f"{plot_name}_histogram": wandb.plot.bar(
                        wandb.Table(dataframe=histogram_df),
                        "Bins",
                        "Percentage",  # wandb plots are horizontal
                        title=plot_name,
                    )
                },
                step=0,
            )

        fig.write_image(f"{path}/{plot_name}.png")
