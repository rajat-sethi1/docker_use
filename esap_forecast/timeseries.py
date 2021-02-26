import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta

class Timeseries:
    """
    Timeseries object for the user-defined uni-variate/multi-variate time series

    Parameters
    ----------
    df [Dataframe]:
        The actual time series as a pandas dataframe with with two columns: 'datetime' and 'value'
    train_test_split [float] (optional):
        The split value for training and testing datasets

    """    
    def __init__(self, passed_df, train_test_split = 0.75):
        if not isinstance(passed_df, pd.DataFrame):
            raise Exception("Passed data must be a pandas dataframe")
        if not len(passed_df) > 0 and not passed_df.shape[1] > 0:
            raise Exception("Passed data cannot be empty")

    @staticmethod
    def get_data(passed_df):
        """
        Args:
            passed_df (dataframe)

        Returns:
            [dataframe]: returns df - the cleaned dataframe
        """        
        df = passed_df.copy()
        value = passed_df['value']
        date = pd.to_datetime(df['datetime'])
        start = date.iloc[0]
        end = date.iloc[-1] + timedelta(minutes=15)  
        start_date = start.strftime("%m/%d/%Y")
        end_date = end.strftime("%m/%d/%Y")
        dates = pd.date_range(start=start_date,end=end_date, freq='15min')
        dates = dates[:-1]
        df = pd.DataFrame( {'load': value, 'datetime': dates })
        df = df.set_index('datetime')
        df = df.replace('?', np.nan)
        df = df.astype(np.float).fillna(method='bfill')
        return df

    def plot_forecasts(self, pr, te):
        """
        Plotting forecasts vs actual points using plotly
        """
        x = [i for i in range(1, len(pr))]

        from plotly import tools
        import plotly
        import plotly.graph_objs as go

        trace0 = go.Scatter(x=x, y=pr, name="Predicted load [in kW]", line=dict(color=("rgb(255, 140, 0)"),))
        trace1 = go.Scatter(x=x, y=te, name="Actual Load [in kW]", line=dict(color=("rgb(100, 149, 237)"),))
       
        # plotly.offline.init_notebook_mode(connected=True)

        data = [trace0, trace1]
        layout = go.Layout(
            xaxis=dict(title="Timesteps", titlefont=dict(family="Courier New, monospace", size=18)),
            yaxis=dict(title="Load [in kW]", titlefont=dict(family="Courier New, monospace", size=18, color="#7f7f7f")),
        )
        fig = go.Figure(data=data, layout=layout)
        plotly.offline.plot(fig, filename="forecast.html")

    def plot_uncertainty(self, predictions, actual, us, ls, us_95, ls_95):
        """
        Plotting uncertainty forecasts using plotly
        """
        x = [i for i in range(1, len(predictions))]
        import plotly
        import plotly.graph_objs as go

        # draw figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=us, mode="lines", name="upper bound (99)"))
        fig.add_trace(go.Scatter(x=x, y=us_95, mode="lines", name="upper bound (95)"))
        fig.add_trace(go.Scatter(x=x, y=predictions, mode="lines", fill="tonexty", name="mean_predictions"))
        # fig.add_trace(go.Scatter(x=x, y=actual, mode="lines", name="actuals"))
        fig.add_trace(go.Scatter(x=x, y=ls_95, mode="lines", fill="tonexty", name="lower bound (95)"))
        fig.add_trace(go.Scatter(x=x, y=ls, mode="lines", fill="tonexty", name="lower bound (99)"))
        plotly.offline.plot(fig, filename="forecast_uncertainty_plot.html")