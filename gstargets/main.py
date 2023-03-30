import plotly.graph_objects as go
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

from gstargets.target_point import getTPs
from gstargets.reversal import getReversalArea
from gstargets.shared import _findWave
from gstargets.config import DIRECTION


def plot_target_points(
    df: pd.DataFrame,
    tradeSide,
    maximumAcceptableBarType3=0,
    thresholdType2=0.5,
    entryPoint=None,
    upWaveNums=[],
    downWaveNums=[],
    nBins=20,
    windowType3=2,
    ignorePercentageUp=20,
    ignorePercentageDown=20,
    zigzagUpThreshold=0.3,
    zigzagDownThreshold=-0.3,
):
    """completely identical inputs to getTPs function
    you can use this function to display the results.
    """
    df.reset_index(inplace=True, drop=True)
    forPlot = df.copy()

    res = getTPs(
        df,
        tradeSide,
        thresholdType2=thresholdType2,
        upWaveNums=upWaveNums,
        downWaveNums=downWaveNums,
        nBins=nBins,
        ignorePercentageUp=ignorePercentageUp,
        ignorePercentageDown=ignorePercentageDown,
        zigzagUpThreshold=zigzagUpThreshold,
        zigzagDownThreshold=zigzagDownThreshold,
        returnPivot=True,
        returnVP=True,
    )
    TPs, res, pivots = res["tps"], res["vp"], res["pivots"]

    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(
        go.Bar(
            x=res.aggregateVolume, y=(res.minPrice + res.maxPrice) / 2, orientation="h"
        ),
        row=1,
        col=2,
    )

    close = forPlot.close
    fig.add_trace(
        go.Scatter(name="close", y=close, mode="lines", marker_color="#D2691E")
    )
    for waveNum in upWaveNums:
        wave = _findWave(pivots, waveNum, DIRECTION.UP)
        fig.add_trace(
            go.Scatter(
                name="close",
                x=np.arange(len(close))[wave[0] : wave[1]],
                y=close[wave[0] : wave[1]],
                mode="lines",
                marker_color="black",
            )
        )
    for waveNum in downWaveNums:
        wave = _findWave(pivots, waveNum, DIRECTION.DOWN)
        fig.add_trace(
            go.Scatter(
                name="close",
                x=np.arange(len(close))[wave[0] : wave[1]],
                y=close[wave[0] : wave[1]],
                mode="lines",
                marker_color="black",
            )
        )
    fig.add_trace(
        go.Scatter(
            name="top",
            x=np.arange(len(close))[pivots == 1],
            y=close[pivots == 1],
            mode="markers",
            marker_color="green",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="bottom",
            x=np.arange(len(close))[pivots == -1],
            y=close[pivots == -1],
            mode="markers",
            marker_color="red",
        )
    )

    for line in TPs:
        fig.add_hline(
            y=line["price"], line_width=3, line_dash="dash", line_color="green"
        )

    fig.show()


def _manual_test_tp():
    # TODO : use yfinance and pytse-client to get plenty of tickers
    # and test automatically
    path = "~/Downloads/data/tickers_data/test.csv"
    df = pd.read_csv(path)

    nBins = 20
    tradeSide = DIRECTION.UP
    downWaveNums = [1]
    zigzagDownThreshold = -0.3
    zigzagUpThreshold = 0.3

    n = 1000
    df = df[-n:]

    df["price"] = (df["high"] + df["low"]) / 2
    df = df[["volume", "price", "close"]]
    plot_target_points(
        df,
        tradeSide,
        nBins=nBins,
        downWaveNums=downWaveNums,
        zigzagUpThreshold=zigzagUpThreshold,
        zigzagDownThreshold=zigzagDownThreshold,
    )


def plot_reversal_area(df, tradeSide, upWaveNums=[], downWaveNums=[]):
    df.reset_index(inplace=True, drop=True)
    forPlot = df.copy()

    res = getReversalArea(
        df,
        tradeSide,
        upWaveNums=upWaveNums,
        downWaveNums=downWaveNums,
        returnPivot=True,
        returnVP=True,
    )

    rev_areas, res, pivots = res["reversal_areas"], res["vp"], res["pivots"]

    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(
        go.Bar(
            x=res.aggregateVolume, y=(res.minPrice + res.maxPrice) / 2, orientation="h"
        ),
        row=1,
        col=2,
    )

    close = forPlot.close
    fig.add_trace(
        go.Scatter(name="close", y=close, mode="lines", marker_color="#D2691E")
    )
    for waveNum in upWaveNums:
        wave = _findWave(pivots, waveNum, DIRECTION.UP)
        fig.add_trace(
            go.Scatter(
                name="close",
                x=np.arange(len(close))[wave[0] : wave[1]],
                y=close[wave[0] : wave[1]],
                mode="lines",
                marker_color="black",
            )
        )
    for waveNum in downWaveNums:
        wave = _findWave(pivots, waveNum, DIRECTION.DOWN)
        fig.add_trace(
            go.Scatter(
                name="close",
                x=np.arange(len(close))[wave[0] : wave[1]],
                y=close[wave[0] : wave[1]],
                mode="lines",
                marker_color="black",
            )
        )
    fig.add_trace(
        go.Scatter(
            name="top",
            x=np.arange(len(close))[pivots == 1],
            y=close[pivots == 1],
            mode="markers",
            marker_color="green",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="bottom",
            x=np.arange(len(close))[pivots == -1],
            y=close[pivots == -1],
            mode="markers",
            marker_color="red",
        )
    )

    for line in rev_areas:
        fig.add_hline(
            y=line["maxPrice"], line_width=3, line_dash="dash", line_color="green"
        )
        fig.add_hline(
            y=line["minPrice"], line_width=3, line_dash="dash", line_color="blue"
        )

    fig.show()


def _manual_test_reversal():
    path = "~/Downloads/data/tickers_data/test.csv"
    df = pd.read_csv(path)

    tradeSide = DIRECTION.UP
    upWaveNums = []
    downWaveNums = [1]

    n = 1000
    df = df[-n:]

    df["price"] = (df["high"] + df["low"]) / 2
    plot_reversal_area(df, tradeSide, upWaveNums=upWaveNums, downWaveNums=downWaveNums)


if __name__ == "__main__":
    _manual_test_tp()
    _manual_test_reversal()
