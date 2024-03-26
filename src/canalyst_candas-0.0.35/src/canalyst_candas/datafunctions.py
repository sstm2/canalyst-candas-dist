import pandas as pd


class DataFunctions(object):
    """
    Helper class for processing model_frame() Pandas dataframe objects.​
    """

    def unstack(
        self, df=pd.DataFrame, ticker="", time_series_name=[], period_type="period_name"
    ):
        """
        A function to unstack a dataframe with period_name as index and either all or specific time series names as column headers.

        Parameters
        ----------
        ticker: str
        time_series_name: list
        period_type: string

        Returns
        ----------
        Pandas DataFrame
        """
        if ticker != "":
            df = df.loc[df["ticker"] == ticker]

        if time_series_name != "":
            if type(time_series_name) is list and len(time_series_name):
                df = df.loc[df["time_series_name"].isin(time_series_name)]

            elif type(time_series_name) is str:
                df = df.loc[df["time_series_name"] == time_series_name]

        dx = (
            df[["ticker", "value", "time_series_name", period_type]]
            .set_index(["ticker", period_type, "time_series_name"])
            .unstack()
        )

        dx.columns = ["_".join(column) for column in dx.columns.to_flat_index()]
        dx = dx.reset_index()
        dx.columns = [c.replace("value_", "") for c in list(dx.columns)]

        return dx

    def time_series_function(
        self,
        df,
        arguments,
        modifier,
        time_series_name="",
        time_series_description="",
    ):
        """
        Apply a specified function to a model_frame() Pandas DataFrame "value" column

        Based on inner join with period name and applying a function to two time series

        Parameters
        ----------
        df: ModelSet.model_frame()
        arguments: list
        modfifier:
        {"add", "subtract", "divide", "multiply","value"}
        time_series_name: str
        time_series_description: str
        """
        first_argument, second_argument = arguments
        df_1 = df.loc[df["time_series_name"] == first_argument]
        df_2 = df.loc[df["time_series_name"] == second_argument]
        df_3 = pd.merge(
            df_1,
            df_2,
            how="inner",
            left_on=["ticker", "period_name"],
            right_on=["ticker", "period_name"],
        )
        df_3 = df_3.dropna()
        df_out = pd.DataFrame()
        df_out["ticker"] = df_3["ticker"]
        df_out["period_name"] = df_3["period_name"]
        if time_series_name == "":
            df_out[
                "time_series_name"
            ] = f"formula_{first_argument}_{modifier}_{second_argument}"
        else:
            df_out["time_series_name"] = str(time_series_name)
        df_out["period_duration_type"] = df_3["period_duration_type_x"]
        df_out["period_start_date"] = df_3["period_start_date_x"]
        df_out["period_end_date"] = df_3["period_end_date_x"]
        df_out["MRFQ"] = df_3["MRFQ_x"]
        df_out["period_name_sorted"] = df_3["period_name_sorted_x"]
        df_out["category"] = "Custom Formula"
        if time_series_description == "":
            df_out[
                "time_series_description"
            ] = f"Formula: {first_argument} {modifier} {second_argument}"
        else:
            df_out["time_series_description"] = str(time_series_description)
        df_out["is_historical"] = df_3["is_historical_x"]
        df_out["is_driver"] = df_3["is_driver_x"]
        df_out["CSIN"] = df_3["CSIN_x"]
        modifier = modifier
        method_name = "func_" + str(modifier)
        method = getattr(
            self,
            method_name,
            lambda: "Invalid function: use add, subtract, divide, or multiply",
        )
        return method(df, df_out, df_3)

    def func_add(self, df, df_out, df_3):
        df_out["value"] = self.df_3["value_x"] + df_3["value_y"]
        return (
            pd.concat([df, df_out])
            .sort_values("ticker", ascending=False)
            .dropna(subset=["value"])
        )

    def func_divide(self, df, df_out, df_3):
        df_out["value"] = df_3["value_x"] / df_3["value_y"]
        return (
            pd.concat([df, df_out])
            .sort_values("ticker", ascending=False)
            .dropna(subset=["value"])
        )

    def func_multiply(self, df, df_out, df_3):
        df_out["value"] = df_3["value_x"] * df_3["value_y"]
        return (
            pd.concat([df, df_out])
            .sort_values("ticker", ascending=False)
            .dropna(subset=["value"])
        )

    def func_subtract(self, df, df_out, df_3):
        df_out["value"] = df_3["value_x"] - df_3["value_y"]
        return (
            pd.concat([df, df_out])
            .sort_values("ticker", ascending=False)
            .dropna(subset=["value"])
        )
