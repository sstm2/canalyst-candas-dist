"""
Model and ModelMap classes

NOTE: separate Model and ModelSet into different files and try to eliminate
circular imports by avoiding direct dependency on other core classes
"""

import re
import os
import os.path
import __main__
import multiprocessing
import webbrowser
from statistics import stdev
from typing import Dict, Iterable, Optional, List

import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import matplotlib.pyplot as plt
from pyvis.network import Network
from pandas.core.frame import DataFrame
from requests.models import Response
from joblib import Parallel, delayed

from canalyst_candas import settings
from canalyst_candas.candas.func_music import FuncMusic
from canalyst_candas.datafunctions import DataFunctions
from canalyst_candas.configuration.config import Config, create_config
from canalyst_candas.exceptions import (
    ScenarioException,
    CSINEmptyException,
)
from canalyst_candas.utils.logger import LogFile
from canalyst_candas.utils.transformations import (
    filter_dataset,
    get_api_headers,
    get_forecast_url,
    map_scenario_urls,
    save_guidance_csv,
)
from canalyst_candas.utils.requests import (
    Getter,
    check_time_series_name,
    get_request,
    create_drivers_dot,
    get_company_info_from_ticker,
    get_data_set_from_mds,
    get_drivers_from_api,
    get_name_index_from_csv,
    get_forecast_url_data,
    get_model_info,
    get_scenario_url_data,
    send_scenario,
)

num_cores = multiprocessing.cpu_count()
plt.style.use("fivethirtyeight")


class ModelMap:
    def __init__(
        self,
        config: Config = None,
        ticker: str = None,
        model: Optional["Model"] = None,
        time_series_name: str = "MO_RIS_REV",
        col_for_labels: str = "time_series_name",
        tree: bool = True,
        common_size_tree: bool = True,
        notebook: bool = True,
        auto_download: bool = True,
        tree_complexity_limit: Optional[
            int
        ] = None,  # model map will not display as a tree if its complexity is above this limit
        common_time_series: Optional[
            Iterable
        ] = None,  # optional list of common features. nodes with this feature will be made triangles
    ) -> None:
        self.config = config or create_config()
        self.s3_client = Getter(config=self.config)
        self.api_headers: Dict[str, str] = get_api_headers(self.config.canalyst_api_key)
        self.log = LogFile()

        if model:
            self.model: Optional["Model"] = model
            self.ticker = self.model.ticker
        elif ticker:
            self.ticker = ticker
            self.model = None
        else:
            raise TypeError("You must pass in either a ticker or model")

        self.time_series = time_series_name
        self.col_for_labels = col_for_labels
        self.tree = tree
        self.common_size_tree = common_size_tree
        self.notebook = notebook
        self.auto_download = auto_download
        self.tree_complexity_limit = tree_complexity_limit
        if common_time_series:
            self.common_time_series: Optional[Iterable] = set(common_time_series)
        else:
            self.common_time_series = common_time_series

        # will be defined when create_model is called
        self.dot_file = None
        self.nodes: list = []
        self.path_distances: dict = {}
        self.complexity: Optional[int] = None
        self.mean_end_node_distance: float = 0
        self.max_end_node_distance = 0
        self.std_dev_node_distance = 0

        # Create DataFrame for precedent/dependent tree, graph of the tree, and a network of that graph (for visualization)
        self.df = self.load_data()
        try:
            self.network = self.create_model()
        except Exception as ex:
            print(f"Modelmap error: {ticker}. Error: {ex}")

        self.fig = None

    def load_data(self) -> pd.DataFrame:
        """
        Helper function to load data from candas.

        Loads model_frame for chosen ticker and creates a dot file.
        """
        if self.model:
            cdm = self.model
        else:
            cdm = Model(config=self.config, ticker=self.ticker)

        df = cdm.model_frame(period_duration_type="fiscal_quarter", n_periods="")

        self.dot_file = create_drivers_dot(
            self.time_series,
            self.api_headers,
            self.ticker,
            self.log,
            self.config,
            self.s3_client,
        )

        return df

    def create_model(self, toggle_drag_nodes=True):
        """
        Create a model of the dot file and optionally download in an html file.
        """
        # read dot file into a graph
        G = nx.drawing.nx_pydot.read_dot(self.dot_file)

        # Create Network from nx graph and turn off physics so dragging is easier
        graph = Network(
            "100%",
            "100%",
            notebook=self.notebook,  # enables displaying in a notebook
            directed=True,  # makes edges arrows
            layout=self.tree,  # Creates tree structure
            bgcolor="#FFFFFF",
        )
        graph.toggle_physics(True)
        graph.from_nx(G)

        # update the complexity of the graph, and if it is greater than the limit, set self.Tree to False
        # won't do anything if tree_complexity_limit wasn't set
        self.complexity = len(graph.nodes)
        if self.tree_complexity_limit and self.complexity > self.tree_complexity_limit:
            self.tree = False

        # model_frame filtered with only needed time_series
        time_series = {}
        for node in graph.nodes:
            id_split = node["id"].split("|")
            if len(id_split) > 1:
                time_series[id_split[0]] = id_split[1]

        df = self.df[self.df["time_series_slug"].isin(time_series)]

        # root node id
        root_id = graph.nodes[0]["id"]

        # column to use for labels
        col_for_labels = self.col_for_labels

        # reformat nodes
        # path lengths between each node
        self.path_distances = dict(nx.all_pairs_shortest_path_length(G))
        num_drivers = 0
        distances = []

        for node in graph.nodes:

            # set level of node based on distance from root
            node["level"] = self.path_distances[root_id][node["id"]]

            # parse node id into time series slug and period name
            # AND get row with the given period and time series slug in DataFrame
            id_split = node["id"].split("|")

            ###### SOLVE FOR MODELS CONTAINING MO.Lastprice (which is not in candas DataFrame)
            # account for time series without a period (last price)
            if len(id_split) < 2:
                node["title"] = node["label"]
                continue
            ######

            slug = id_split[0]

            period = id_split[1]
            rows = df[df["period_name"] == period]
            try:
                row = rows[rows["time_series_slug"] == slug].iloc[0]
            except:
                print("ModelMap Error: " + str(node))
                print("period: " + period)
                continue

            # format colors of nodes and update data
            if row["is_driver"]:
                color = "rgba(200, 0, 0, 0.75)"

                ########## STATS #############
                # get dist from root node, add to mean and find new max
                distance_from_root = self.path_distances[root_id][node["id"]]
                distances.append(distance_from_root)
                self.mean_end_node_distance += distance_from_root
                if distance_from_root > self.max_end_node_distance:
                    self.max_end_node_distance = distance_from_root
                num_drivers += 1
                ##########
            else:
                color = "rgba(0, 0, 200, 0.75)"
            node["color"] = color

            # make the node a triangle if the it is one of the common_time_series
            if (
                self.common_time_series
                and row["time_series_name"] in self.common_time_series
            ):
                node["borderWidth"] = 5
                node["shape"] = "triangle"

            # store description, mo_name, ticker, and driver status of time series in node
            node["description"] = str(row["time_series_description"])
            node["time_series_name"] = str(row["time_series_name"])
            node["ticker"] = str(row["ticker"])
            node["is_driver"] = str(row["is_driver"])

            # set the label to be whatever column the user decided to take labels from
            label = str(row[col_for_labels])
            # Format the label to be max n characters per line
            n = 13  # characters per line
            new_label = "\n".join([label[i : i + n] for i in range(0, len(label), n)])
            node["label"] = new_label

            # add value and units and ismm (time_series_description) attributes to node for future use
            node["amount"] = row["value"]
            node["units"] = row["unit_type"]
            node["ismm"] = ", mm" in row["time_series_description"]

            # Add value to node label
            value = ""
            if node["units"] == "currency":
                value = ":\n{}{:,.2f}".format(row["unit_symbol"], row["value"])
            elif node["units"] == "percentage":
                value = ":\n{:.2f}{}".format(row["value"], row["unit_symbol"])
            # elif node["units"] == "count":
            #     value = ":\n{:.2f} {}".format(row["value"], row["unit_symbol"])
            # elif node["units"] == "ratio":
            #     value = ":\n{:.2f} {}".format(row["value"], row["unit_symbol"])
            # elif node["units"] == "time":
            #     value = ":\n{:.2f} {}".format(row["value"], row["unit_symbol"])
            else:
                value = ":\n{:.2f} {}".format(row["value"], row["unit_symbol"])
            node["label"] += value

            # title is label without newlines
            node["title"] = node["label"].replace("\n", "")

        # reformat edges
        for edge in graph.edges:
            temp = edge["from"]
            edge["from"] = edge["to"]
            edge["to"] = temp
            # label the edge with percentages
            if self.common_size_tree:
                precedent = graph.get_node(edge["from"])
                dependent = graph.get_node(edge["to"])

                # WORKAROUND FOR MO.Lastprice ##################
                if "|" not in precedent["id"] or "|" not in dependent["id"]:
                    continue
                ################################################

                if (
                    precedent["units"] == "currency"
                    and precedent["ismm"]
                    and dependent["ismm"]
                    and precedent["level"] != dependent["level"]
                ):
                    if np.isnan(precedent["amount"]):
                        pct = 0
                    else:
                        if dependent["amount"].sum() > 0:
                            pct = precedent["amount"] / dependent["amount"] * 100
                        else:
                            pct = 0
                    edge["label"] = "{:.2f}%".format(pct)

        # reformat tree so desired time series is at the root and it is a different color
        root = graph.get_node(root_id)
        root.update({"color": "rgba(0, 200, 0, 0.75)", "size": 40})

        if self.notebook:
            graph.width, graph.height = "1000px", "1000px"

        # toggles node dragging and disables physics
        graph.toggle_drag_nodes(toggle_drag_nodes)
        graph.toggle_physics(False)

        # create path for model in models folder adjacent to this file
        self.model_path = os.path.join(
            self.config.default_dir,
            f"{self.ticker.split()[0]}_{self.time_series}_model_map.html",
        )

        if self.auto_download:
            graph.write_html(self.model_path, notebook=self.notebook)

        self.nodes = graph.nodes
        self.edges = graph.edges

        # update stats
        self.mean_end_node_distance /= num_drivers
        self.std_dev_node_distance = stdev(distances)

        return graph

    def show(self):
        """
        Shows the model map
        """
        model_map_name = f"{self.ticker.split()[0]}_{self.time_series}_model_map.html"
        try:
            if self.notebook:
                return self.network.show(model_map_name)

            else:
                self.network.write_html(self.model_path)
                model_map_path = self.model_path.lstrip("/")
                webbrowser.open(f"file:///{model_map_path}")

        except Exception as e:
            print(f"Error in showing model map: {e}")

    def create_node_df(self):
        """
        Creates a DataFrame of each node's MO_name and distance from root
        """
        graph = self.network

        # root node id
        root_id = graph.nodes[0]["id"]

        # columns for DataFrame
        tickers = []
        time_series_names = []
        distances_to_root = []
        is_driver_col = []
        for node in graph.nodes:
            tickers.append(node["ticker"])
            time_series_names.append(node["time_series_name"])
            distances_to_root.append(self.path_distances[root_id][node["id"]])
            is_driver_col.append(node["is_driver"])

        node_df = pd.DataFrame(
            {
                "ticker": tickers,
                "time_series_name": time_series_names,
                "distance_to_root": distances_to_root,
                "is_driver": is_driver_col,
            }
        )

        return node_df

    def list_time_series(self, search=""):
        """
        Lists out all the nodes
        """
        list_out: List[str] = []
        if self.nodes:
            for node in self.nodes:
                if re.match(f".*{search}.*", node["title"]):
                    list_out.append(node["title"].split(":")[0])
        return list_out

    def time_series_names(self, search=""):
        """
        (Obsolete) Lists out all the nodes

        This method is kept for backwards-compatibility. New code should use list_time_series instead.
        """
        list_out = []
        if self.nodes:
            for node in self.nodes:
                if re.match(f".*{search}.*", node["title"]):
                    list_out.append(node["title"].split(":")[0])
        return list_out

    def create_time_series_chart(self, time_series_name):
        """
        Create a chart of given time series
        """
        # get needed data from DataFrame
        df = self.df[self.df["time_series_name"] == time_series_name]

        if df.shape[0] == 0:
            print("Time series not found")
            return None

        df = df[df["period_name"].str.contains("Q")].sort_values("period_end_date")
        # use subset=['value'] to only drop rows with a null value
        df = df.dropna(subset=["value"])
        # NEED TO ACCOUNT FOR EMPTY DataFrameS
        row1 = df.iloc[0]
        title = row1["time_series_description"]
        xlabel = "Fiscal Quarter"
        ylabel = row1["time_series_description"]
        # plot data
        fig = px.line(
            df,
            x="period_name",
            y="value",
            title=title,
            labels={"period_name": xlabel, "value": ylabel},
        )
        return fig

    def show_time_series_chart(self, time_series_name):
        """
        Shows a chart of the given time series
        """
        fig = self.create_time_series_chart(time_series_name)
        fig.show()
        return


class Model:
    # functions which are CamelCase will become Classes at some point
    def __init__(
        self,
        ticker: str,
        config: Config = None,
        force: bool = False,
        extract_drivers: bool = True,
        company_info: bool = True,
        file_type: str = "parquet",
    ):
        self.file_type = file_type
        self.config = config or create_config()
        self.ticker = ticker
        self.force = force
        self.company_info = company_info
        self.extract_drivers = extract_drivers
        self.log = LogFile()

        self.api_headers = get_api_headers(self.config.canalyst_api_key)
        self.gt = Getter(self.config)

        self.set_new_uuid()
        self.scenario_url_map: Dict[str, str] = {}

        self.csin: str = ""
        self.latest_version: str = ""
        self._revenue_drivers = None

        if self.force == True:
            print("model_frame")

        if self.company_info == True:
            try:
                self.csin, self.latest_version = self.get_company_info()
            except Exception as ex:
                print(f"Model Error: {self.ticker}. Error: {ex}")
                return
        try:
            self.get_model_frame()

            if self.extract_drivers == True:
                self.apply_drivers()

            if self.force == True:
                print("model_drivers")

            if self.force == True:
                print("guidance")
                self.create_guidance_csv()
        except Exception as ex:
            print(f"Model Error: {self.ticker}. Error: {ex}")
            return

    def time_series_formula(
        self,
        arguments=[],
        modifier="",
        time_series_name="",
        time_series_description="",
    ):
        df = self.model_frame()
        DF = DataFunctions()
        df = DF.time_series_function(
            df,
            arguments,
            modifier,
            time_series_name,
            time_series_description,
        )
        self._model_frame = df  # modify in place
        return

    def model_info(self):
        self.return_string, self.model_info = get_model_info(
            auth_headers=self.api_headers, log=self.log, ticker=self.ticker
        )
        df = pd.DataFrame(self.model_info).T.reset_index()
        df.columns = [
            "ticker",
            "CSIN",
            "name",
            "model_version",
            "historical_periods",
            "update_type",
            "publish_date",
        ]
        self.model_info = df
        return self.model_info

    def key_driver_map(self, time_series_name="MO_RIS_REV"):
        """
        Returns a dataframe of key drivers for a given time series

        Parameters

        time_series_name : str, default "MO_RIS_REV"
        """
        # defaulting to MO_RIS_REV, but you could choose another time_series_name as a starting point like MO_RIS_EBIT
        model_map = self.create_model_map(
            time_series_name=time_series_name,
            col_for_labels="time_series_name",
            notebook=True,
        )

        # list all the nodes in the model map
        model_map_time_series_list = model_map.list_time_series()

        # make it a dataframe because I'm a reformed R user
        df = pd.DataFrame(model_map_time_series_list)

        # columns for joining
        df.columns = ["time_series_name"]
        df["ticker"] = self.ticker

        # create a model frame with only key drivers and just the most recent quarter
        df_drivers = self.model_frame(
            is_driver=True, period_duration_type="fiscal_quarter", mrq=True
        )
        # merge (inner join)
        df = pd.merge(
            df,
            df_drivers,
            how="inner",
            left_on=["ticker", "time_series_name"],
            right_on=["ticker", "time_series_name"],
        )

        return df

    def plot_guidance(self, time_series_name):
        df_guidance = self.guidance()
        df_guidance["time_series_name"] = time_series_name
        df_guidance["period_name"] = df_guidance["Fiscal Period"]
        df_guidance["Actual"] = df_guidance["Output"]
        df_guidance = df_guidance[df_guidance["Item Name"] == time_series_name][
            [
                "time_series_name",
                "period_name",
                "Low",
                "High",
                "Mid",
                "Type.1",
                "Actual",
            ]
        ]
        df_guidance = (
            df_guidance.groupby(["time_series_name", "period_name"])
            .first()
            .reset_index()
        )

        i_low = df_guidance["Low"].sum()
        i_mid = df_guidance["Mid"].sum()
        i_high = df_guidance["High"].sum()

        if len({i_low, i_mid, i_high}) == 1:
            i_three = True
        else:
            i_three = False

        colors = {
            "red": "#ff207c",
            "grey": "#C3C2C3",
            "blue": "#00838F",
            "orange": "#ffa320",
            "green": "#00ec8b",
        }

        plt.rc("figure", figsize=(12, 9))
        plt.title(time_series_name)
        plt.plot(
            df_guidance["period_name"],
            df_guidance["Actual"],
            label="Actual",
            color=colors["blue"],
            linewidth=2,
        )
        plt.plot(
            df_guidance["period_name"],
            df_guidance["Mid"],
            label="Mid",
            color=colors["orange"],
            linewidth=2,
        )

        if i_three == False:
            plt.plot(
                df_guidance["period_name"],
                df_guidance["Low"],
                label="Low",
                color=colors["red"],
                linewidth=2,
            )
            plt.plot(
                df_guidance["period_name"],
                df_guidance["High"],
                label="High",
                color=colors["green"],
                linewidth=2,
            )

        plt.legend()
        plt.show()
        return

    def get_most_recent_model_date(self):
        """
        Get published date for latest workbook model
        """
        csin = self.csin
        auth_headers = self.api_headers
        mds_host = self.config.mds_host
        wp_host = self.config.wp_host

        from python_graphql_client import GraphqlClient

        client = GraphqlClient(endpoint=f"{wp_host}/model-workbooks")

        # Create the query string and variables required for the request.
        query = """
        query driversWorksheetByCSIN($csin: ID!) {
            modelSeries(id: $csin) {
            latestModel {
                id
                name
                publishedAt
                variantsByDimensions(
                    driversWorksheets: [STANDARD_FCF],
                    periodOrder: [CHRONOLOGICAL],
                ) {
                id
                downloadUrl
                variantDimensions {
                    driversWorksheets
                    periodOrder
                }
                }
            }
            }
        }
        """
        variables = {"csin": csin}

        # Synchronous request
        data = client.execute(
            query=query,
            variables=variables,
            headers=auth_headers,
            verify=self.config.verify_ssl,
        )
        url = data["data"]["modelSeries"]["latestModel"]["variantsByDimensions"][0][
            "downloadUrl"
        ]
        file_ticker = self.ticker.split(" ")[0]
        date_name = data["data"]["modelSeries"]["latestModel"]["publishedAt"]
        date_name = date_name.split("T")[0]
        return date_name

    def get_excel_model_name(self):
        """
        Get the model workbook's file name
        """
        csin = self.csin
        auth_headers = self.api_headers
        mds_host = self.config.mds_host
        wp_host = self.config.wp_host

        from python_graphql_client import GraphqlClient

        client = GraphqlClient(endpoint=f"{wp_host}/model-workbooks")

        # Create the query string and variables required for the request.
        query = """
        query driversWorksheetByCSIN($csin: ID!) {
            modelSeries(id: $csin) {
            latestModel {
                id
                name
                publishedAt
                variantsByDimensions(
                    driversWorksheets: [STANDARD_FCF],
                    periodOrder: [CHRONOLOGICAL],
                ) {
                id
                downloadUrl
                variantDimensions {
                    driversWorksheets
                    periodOrder
                }
                }
            }
            }
        }
        """
        variables = {"csin": csin}

        # Synchronous request
        data = client.execute(
            query=query,
            variables=variables,
            headers=auth_headers,
            verify=self.config.verify_ssl,
        )
        url = data["data"]["modelSeries"]["latestModel"]["variantsByDimensions"][0][
            "downloadUrl"
        ]
        file_ticker = self.ticker.split(" ")[0]
        file_name = data["data"]["modelSeries"]["latestModel"]["name"]
        return file_name

    def forecast_frame(
        self, time_series_name, n_periods, function_name="value", function_value=""
    ):
        df = self.model_frame(
            [time_series_name],
            period_duration_type="fiscal_quarter",
            is_historical=False,
            n_periods=n_periods,
        )

        if function_name != "value":
            d = FuncMusic()
            df["new_value"] = df.apply(
                lambda row: d.apply_function(
                    row["value"], modifier=function_name, argument=function_value
                ),
                axis=1,
            )
        else:
            df["new_value"] = function_value

        df = df[
            [
                "ticker",
                "period_name",
                "time_series_name",
                "time_series_description",
                "value",
                "new_value",
                "period_end_date",
            ]
        ]

        return df

    def create_time_series_chart(self, time_series_name, historical=False):  # MODEL
        # get needed data from DataFrame
        df = self._model_frame[
            self._model_frame["time_series_name"] == time_series_name
        ]
        if df.shape[0] == 0:
            print("Time series not found")
            return None

        if historical:
            df = df.loc[df["is_historical"] == historical]

        df = df[df["period_name"].str.contains("Q")].sort_values("period_end_date")
        # use subset=['value'] to only drop rows with a null value
        df = df.dropna(subset=["value"])
        # NEED TO ACCOUNT FOR EMPTY DataFrameS
        row1 = df.iloc[0]
        title = self.ticker + " " + row1["time_series_description"]
        xlabel = "Fiscal Quarter"
        ylabel = row1["time_series_description"]

        colors = {
            "red": "#ff207c",
            "grey": "#C3C2C3",
            "blue": "#00838F",
            "orange": "#ffa320",
            "green": "#00ec8b",
        }

        plt.rc("figure", figsize=(12, 9))
        plt.title(time_series_name)
        plt.plot(
            df["period_name"],
            df["value"],
            label=ylabel,
            color=colors["blue"],
            linewidth=2,
        )
        plt.axvline(x=self.mrq()["period_name"][0], color="black")
        plt.xticks(rotation=90)
        plt.legend()
        plt.show()

        return

    def create_model_map(
        self,
        time_series_name: str = "MO_RIS_REV",
        col_for_labels: str = "time_series_name",
        tree: bool = True,
        notebook: bool = True,
        common_time_series_names: Optional[Iterable] = None,
    ) -> ModelMap:
        """
        Create a model map from this model, rooted at the specified time series

        Defaults to a model map rooted at MO_RIS_REV if no time series name is provided
        """
        mm = ModelMap(
            config=self.config,
            model=self,
            time_series_name=time_series_name,
            col_for_labels=col_for_labels,
            tree=tree,
            notebook=notebook,
            common_time_series=common_time_series_names,
        )
        self.model_map = mm
        return mm

    def get_params(self, search_term=""):
        df = self.default_df
        if search_term != "":
            search_term = "(?i)" + search_term
            df1 = df[df["time_series_description"].str.contains(search_term)]
            df2 = df[df["time_series_name"].str.contains(search_term)]
            df = pd.concat([df1, df2])
            df = (
                df.groupby(["time_series_name", "time_series_description"])
                .first()
                .reset_index()
            )
        df = df.sort_values("time_series_name")
        return df[["time_series_name", "time_series_description"]]

    def set_params(self, list_params=[]):
        self.set_new_uuid()
        scenario_name = self.uuid
        list_changes = []
        for param in list_params:
            list_changes.append(
                {
                    "time_series": param["feature_name"],
                    "period": param["feature_period"],
                    "value_expression": {
                        "type": "literal",
                        "value": param["feature_value"],
                    },
                }
            )

        data = {"changes": list_changes, "name": scenario_name}

        response = send_scenario(
            self.ticker,
            data,
            self.api_headers,
            self.log,
            self.csin,
            self.latest_version,
            self.config.mds_host,
            self.config.verify_ssl,
            self.config.proxies,
        )

        self.record_scenario_url(response)

        return

    def show_model_map(
        self,
        time_series="MO_RIS_REV",
        tree=True,
        notebook=True,
        common_time_series=None,
    ):
        """
        Display model_map
        """
        mm = ModelMap(
            config=self.config,
            model=self,
            time_series_name=time_series,
            tree=tree,
            notebook=notebook,
            common_time_series=common_time_series,
        )
        self.model_map = mm
        return mm.show()

    def create_guidance_csv(self):
        """
        Write a CSV file containing the model's guidance data

        Parses guidance from the model's Excel workbook
        """
        file_ticker = self.ticker.split(" ")[0]
        path_name = f"{self.config.default_dir}/DATA/{file_ticker}/"
        os.makedirs(path_name, exist_ok=True)

        files = os.listdir(path_name)

        for filename in files:

            if "xlsx" in str(filename):
                try:
                    print("read excel for guidance")
                    df = pd.read_excel(
                        open(f"{path_name}" + filename, "rb"),
                        index_col=False,
                        sheet_name="Guidance",
                        engine="openpyxl",
                    )
                except:
                    print("Read excel for guidance error")
                    return
                try:
                    save_guidance_csv(df, self.ticker, path_name)
                    return
                except:
                    print("Save guidance csv error")

    def set_new_uuid(self):
        import uuid

        self.uuid = str(uuid.uuid4())[:8]

    def mrq(self):
        df = self._model_frame
        df = df[df["is_historical"] == True]
        df = df[~df["period_name"].str.contains("FY")]
        df = df.sort_values("period_end_date", ascending=False)
        return pd.DataFrame(
            df.iloc[0][["period_name", "period_end_date"]]
        ).T.reset_index()[["period_name", "period_end_date"]]

    def guidance(self):
        df = self.gt.get_file(self.ticker, "guidance", file_type="csv")
        df = df.dropna(how="all")
        self._guidance = df
        return df

    def get_drivers(self):
        """
        Gets the associated list of drivers for the model.
        """
        drivers = get_drivers_from_api(
            self.config.mds_host,
            self.csin,
            self.latest_version,
            self.api_headers,
            self.log,
            self.config.verify_ssl,
            self.config.proxies,
        )

        df = pd.json_normalize(drivers)
        if not df.empty:
            df_driver_array = pd.DataFrame(df["time_series.names"])
        else:
            df_driver_array = pd.DataFrame(columns=["time_series.names"])

        df_drivers = df_driver_array.explode("time_series.names")
        df = df_drivers.groupby("time_series.names").first().reset_index()
        df.columns = ["time_series_name"]

        self._model_drivers = df

    def get_company_info(self):
        """
        Get CSIN and latest version for a ticker
        """
        return get_company_info_from_ticker(
            ticker=self.ticker,
            auth_headers=self.api_headers,
            logger=self.log,
            mds_host=self.config.mds_host,
            verify_ssl=self.config.verify_ssl,
            proxies=self.config.proxies,
        )

    def get_name_index(self, force=False):

        self.df_name_index = get_name_index_from_csv(self.ticker, self.config)

        ticker = self.ticker.split(" ")[0]

        self.df_name_index.to_csv(
            f"{self.config.default_dir}/DATA/{ticker}/{self.ticker}_name_index.csv",
            index=False,
        )
        return

    def revenue_drivers(self, search_term="", is_driver=False):
        pd.set_option("display.max_colwidth", None)
        df = self._model_frame
        df_only_drivers = self._revenue_drivers
        df = pd.merge(
            df,
            df_only_drivers,
            how="inner",
            left_on="time_series_name",
            right_on="time_series_name_dependent",
        )
        df1 = df[
            df["time_series_description"].str.contains(
                search_term, flags=re.IGNORECASE, regex=True
            )
        ]
        df2 = df[
            df["time_series_name"].str.contains(
                search_term, flags=re.IGNORECASE, regex=True
            )
        ]
        df = pd.concat([df1, df2])
        df = df.dropna()
        df = (
            df.groupby(["time_series_name", "time_series_description"])
            .first()
            .reset_index()
        )
        df = df.drop(
            columns=[
                "period_duration_type",
                "category_type_slug",
                "time_series_slug",
                "category_type_name",
                "category_slug",
                "is_historical",
                "value",
                "period_start_date",
                "period_end_date",
                "period_name",
                "time_series_name",
                "time_series_description",
            ]
        )
        df = df.sort_values("index")
        df = df.reset_index()
        df = df.drop(columns=["index", "level_0"])
        if is_driver == True:
            df = df.loc[df["is_driver"] == True]
            return df
        return df

    def model_drivers(self, search_term=""):
        mrq = self.mrq()["period_name"][0]
        df = self._model_drivers
        df2 = self._model_frame
        df2 = df2.loc[df2["period_name"] == mrq]
        df = pd.merge(
            df,
            df2,
            how="inner",
            left_on="time_series_name",
            right_on="time_series_name",
        )[
            ["name_index", "category", "time_series_name", "time_series_description"]
        ].sort_values(
            "name_index", ascending=True
        )

        if search_term != "":
            df = df[
                df["time_series_name"].str.contains(
                    search_term, flags=re.IGNORECASE, regex=True
                )
            ]
        return df

    # MODEL model_frame
    def model_frame(
        self,
        time_series_name="",
        period_name="",
        is_driver="",
        pivot=False,
        mrq=False,
        period_duration_type="",
        is_historical="",
        n_periods=36,
        mrq_notation=False,
        unit_type="",
        category="",
        warning=True,
    ):
        if self.csin == "":
            raise CSINEmptyException("Error: CSIN not found")

        if time_series_name != "":
            self._model_frame = check_time_series_name(
                self._model_frame,
                time_series_name,
                self.api_headers,
                self.log,
                self.config.verify_ssl,
            )

        return filter_dataset(
            self._model_frame,
            time_series_name,
            period_name,
            is_driver,
            pivot,
            mrq,
            period_duration_type,
            is_historical,
            n_periods,
            mrq_notation,
            unit_type,
            category,
            warning,
        )

    def apply_drivers(self):

        self.get_drivers()
        df = self._model_drivers
        df = df.assign(is_driver=True)

        df2 = pd.merge(
            self._model_frame,
            df,
            how="outer",
            left_on="time_series_name",
            right_on="time_series_name",
        )
        df2 = df2[df2["ticker"].notna()]
        df2[["is_driver"]] = df2[["is_driver"]].fillna(value=False)
        self._model_frame: DataFrame = df2

        df_index = get_data_set_from_mds(
            settings.BulkDataKeys.NAME_INDEX,
            self.file_type,
            self.csin,
            self.latest_version,
            self.api_headers,
            self.log,
            settings.MDS_HOST,
            self.config.verify_ssl,
            self.config.proxies,
        )

        if df_index is not None:
            df_index.columns = ["time_series_name", "name_index"]
        else:
            mrq = self._model_frame["MRFQ"].loc[0]
            df_index = self._model_frame.loc[self._model_frame["period_name"] == mrq][
                ["time_series_name"]
            ]
            df_index["name_index"] = np.arange(len(df_index))

        self._model_frame = pd.merge(
            self._model_frame,
            df_index,
            how="outer",
            left_on="time_series_name",
            right_on="time_series_name",
        )

        self._model_frame = self._model_frame[self._model_frame["period_name"].notna()]
        return

    def get_model_frame(self):
        df_hist = self.historical_data_frame()

        df_hist = df_hist.assign(is_historical=True)
        df_hist = df_hist[~df_hist["period_name"].isna()]

        mrq = df_hist["period_name"].iloc[0]
        pd.options.mode.chained_assignment = None

        df_fwd = self.forward_data_frame()

        df_fwd = df_fwd.assign(is_historical=False)
        df_fwd = df_fwd[~df_fwd["period_name"].isna()]

        df_concat = pd.concat([df_hist, df_fwd]).sort_values("period_name")

        self._model_frame = df_concat

        mrq = self.mrq()["period_name"][0]
        self._model_frame["MRFQ"] = [mrq for i in range(len(self._model_frame))]

        self._model_frame["period_name_sorted"] = np.where(
            self._model_frame["period_name"].str.contains("-"),
            self._model_frame["period_name"].str.split("-").str[1]
            + self._model_frame["period_name"].str.split("-").str[0],
            self._model_frame["period_name"].str[2:6]
            + self._model_frame["period_name"].str[0:2],
        )
        self._model_frame["CSIN"] = self.csin
        return

    def feature_search(self, search_term=""):
        if search_term != "":
            df = self._model_frame
            df1 = df[
                df["time_series_description"].str.contains(
                    search_term, flags=re.IGNORECASE, regex=True
                )
            ]
            df2 = df[
                df["time_series_name"].str.contains(
                    search_term, flags=re.IGNORECASE, regex=True
                )
            ]
            df = pd.concat([df1, df2])
            df = (
                df.groupby(["time_series_name", "time_series_description"])
                .first()
                .reset_index()
            )
            df = df[["time_series_name", "time_series_description"]]
        return df

    def driver_search(self, search_term=""):
        """
        Search modelset drivers
        """
        df = self._model_drivers
        if search_term != "":
            df = self._model_frame
            df_only_drivers = self._model_drivers
            df_drivers = pd.merge(
                df,
                df_only_drivers,
                how="inner",
                left_on="time_series_name",
                right_on="time_series_name",
            )
            df1 = df[
                df["time_series_description"].str.contains(
                    search_term, flags=re.IGNORECASE, regex=True
                )
            ]
            df2 = df[
                df["time_series_name"].str.contains(
                    search_term, flags=re.IGNORECASE, regex=True
                )
            ]
            df = pd.concat([df1, df2])
            df = (
                df.groupby(["time_series_name", "time_series_description"])
                .first()
                .reset_index()
            )
            df = df[["time_series_name", "time_series_description"]]
        return df

    def describe_drivers(self, filter_string=""):
        """
        Get a summary DataFrame of the model's drivers
        """
        # if self.describe_drivers is not None:
        #    return self.describe_drivers
        # get the historical DataFrame from candas
        df = self.historical_data_frame()

        df_only_drivers = self._model_drivers
        # merge the drivers and the historical data... and return a description of the values for each
        df_drivers = pd.merge(
            df,
            df_only_drivers,
            how="inner",
            left_on="time_series_name",
            right_on="time_series_name",
        )
        df_drivers_summary = (
            df_drivers.groupby(["time_series_name"]).describe().reset_index()
        )
        if filter_string != "":
            self._describe_drivers = df_drivers_summary
            return df_drivers_summary[
                df_drivers_summary["time_series_name"].str.contains(
                    filter_string, case=False
                )
            ]
        self._describe_drivers = df_drivers_summary
        return df_drivers_summary

    def historical_data_frame(self):  # if cache = true save locally?
        """
        Get a DataFrame of the model's historical data
        """
        df = get_data_set_from_mds(
            settings.BulkDataKeys.HISTORICAL_DATA,
            self.file_type,
            self.csin,
            self.latest_version,
            self.api_headers,
            self.log,
            self.config.mds_host,
            self.config.verify_ssl,
            self.config.proxies,
        )
        if df is not None:
            return df

    def summary(self, filter_term=""):
        pd.set_option("mode.chained_assignment", None)
        pd.set_option("display.float_format", lambda x: "%.5f" % x)

        df = pd.merge(
            self.model_frame(
                period_duration_type="fiscal_quarter",
                is_historical=False,
                warning=False,
            ),
            self.scenario_df,
            how="inner",
            left_on=[
                "ticker",
                "period_name",
                "time_series_name",
                "time_series_description",
            ],
            right_on=[
                "ticker",
                "period_name",
                "time_series_name",
                "time_series_description",
            ],
        )[
            [
                "ticker",
                "period_name",
                "time_series_name",
                "time_series_description",
                "value_x",
                "value_y",
            ]
        ]
        df.columns = [
            "ticker",
            "period_name",
            "time_series_name",
            "time_series_description",
            "default",
            "scenario",
        ]
        df["diff"] = df["scenario"].astype(float) / df["default"].astype(float)
        if filter_term != "":
            df = df[df["time_series_name"].str.contains(filter_term, case=False)]
        return df

    def forward_data_frame(self):
        if self.force == False:
            df = get_data_set_from_mds(
                settings.BulkDataKeys.FORECAST_DATA,
                self.file_type,
                self.csin,
                self.latest_version,
                self.api_headers,
                self.log,
                self.config.mds_host,
                self.config.verify_ssl,
                self.config.proxies,
            )
            if df is not None:
                self.default_df = df
                return df

        file_ticker = self.ticker.split(" ")[0]
        path_name = f"{self.config.default_dir}/DATA/{file_ticker}/"
        file_name = f"{path_name}{self.ticker}_forecast_data.csv"

        if not os.path.exists(path_name):
            os.makedirs(path_name)

        if self.csin == "":
            self.csin, self.latest_version = get_company_info_from_ticker(
                self.ticker,
                self.api_headers,
                self.log,
                self.config.mds_host,
                self.config.verify_ssl,
                self.config.proxies,
            )

        url = get_forecast_url(self.csin, self.latest_version, self.config.mds_host)

        res = get_request(
            url, self.api_headers, self.log, self.config.verify_ssl, self.config.proxies
        )

        list_out = []

        for res_dict in res.json()["results"]:
            df = get_forecast_url_data(
                res_dict,
                self.ticker,
                self.api_headers,
                self.log,
                self.config.verify_ssl,
                self.config.proxies,
            )
            list_out.append(df)

        self.default_df = pd.concat(list_out)
        self.default_df.to_csv(file_name, index=False)

        return df

    def model_fit(self, time_series_name=""):

        if self.csin == "":
            self.csin, self.latest_version = get_company_info_from_ticker(
                self.ticker,
                auth_headers=self.api_headers,
                logger=self.log,
                mds_host=self.config.mds_host,
                verify_ssl=self.config.verify_ssl,
                proxies=self.config.proxies,
            )

        url = get_forecast_url(self.csin, self.latest_version, self.config.mds_host)

        scenario_url = (
            f"{self.config.mds_host}/"
            f"{settings.SCENARIO_URL.format(csin=self.csin, version=self.latest_version)}"
        ) + "?page_size=200"

        if self.scenario_url_map.get(self.uuid) is None:
            scenario_response = get_request(
                scenario_url,
                headers=self.api_headers,
                logger=self.log,
                verify_ssl=self.config.verify_ssl,
                proxies=self.config.proxies,
            )
            scenario_json = scenario_response.json()
            scenario_id_url = map_scenario_urls(scenario_json).get(self.uuid)
        else:
            scenario_id_url = self.scenario_url_map[self.uuid]

        if scenario_id_url is None:

            print("Scenario ID for " + scenario_url + " is None.")

            return

        print(self.ticker + " scenario_id_url: " + str(scenario_id_url))

        if time_series_name != "":

            url = scenario_id_url + "time-series/?name=" + time_series_name

            res_loop = get_request(
                url,
                self.api_headers,
                self.log,
                self.config.verify_ssl,
                self.config.proxies,
            )
            url = res_loop.json()["results"][0]["self"]
            res_loop = get_request(
                url,
                self.api_headers,
                self.log,
                self.config.verify_ssl,
                self.config.proxies,
            )
            url = res_loop.json()["forecast_data_points"]
            res_loop = get_request(
                url,
                self.api_headers,
                self.log,
                self.config.verify_ssl,
                self.config.proxies,
            )
            res_data = res_loop.json()["results"]
            dict_out = {}
            list_out = []
            for res_data_dict in res_data:
                dict_out["time_series_slug"] = res_data_dict["time_series"]["slug"]
                dict_out["time_series_name"] = res_data_dict["time_series"]["names"][0]
                dict_out["time_series_description"] = res_data_dict["time_series"][
                    "description"
                ]
                dict_out["category_slug"] = res_data_dict["time_series"]["category"][
                    "slug"
                ]
                dict_out["category_type_slug"] = res_data_dict["time_series"][
                    "category"
                ]["type"]["slug"]
                dict_out["category_type_name"] = res_data_dict["time_series"][
                    "category"
                ]["type"]["name"]
                dict_out["unit_description"] = res_data_dict["time_series"]["unit"][
                    "description"
                ]
                dict_out["unit_symbol"] = res_data_dict["time_series"]["unit"]["symbol"]
                dict_out["period_name"] = res_data_dict["period"]["name"]
                dict_out["period_duration_type"] = res_data_dict["period"][
                    "period_duration_type"
                ]
                dict_out["period_start_date"] = res_data_dict["period"]["start_date"]
                dict_out["period_end_date"] = res_data_dict["period"]["end_date"]
                dict_out["value"] = res_data_dict["value"]
                dict_out["ticker"] = self.ticker
                df = pd.DataFrame.from_dict(dict_out, orient="index").T
                list_out.append(df)
            self.scenario_df: DataFrame = pd.concat(list_out)
            return

        scenario_id_url = scenario_id_url + "forecast-periods/"

        scenario_response = get_request(
            scenario_id_url,
            headers=self.api_headers,
            logger=self.log,
            verify_ssl=self.config.verify_ssl,
            proxies=self.config.proxies,
        )
        all_df = []

        all_df = Parallel(n_jobs=num_cores)(
            delayed(get_scenario_url_data)(
                res_dict,
                self.ticker,
                self.api_headers,
                verify_ssl=self.config.verify_ssl,
                proxies=self.config.proxies,
            )
            for res_dict in scenario_response.json()["results"]
        )
        self.scenario_df = pd.concat(all_df)
        print("Done")
        return

    def record_scenario_url(self, scenario_response: Optional[Response]) -> None:

        if scenario_response is None:
            return None

        if not scenario_response.ok:
            raise ScenarioException(scenario_response.status_code)

        """
        Extract scenario name and location from the response and add to scenario url dictionary
        """
        scenario_name = self.uuid
        scenario_url = scenario_response.headers["Location"]

        if self.scenario_url_map.get(scenario_name) is None:
            self.scenario_url_map[scenario_name] = scenario_url
