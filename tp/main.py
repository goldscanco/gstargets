from zigzag import *
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import volprofile as vp
from tp.config import DIRECTION

def _findWave(pivots, waveNum=1):
    indices = np.where(pivots != 0)[0]
    return indices[-waveNum - 1], indices[-waveNum]

def _getTPsIdx(histogram, trend, ignorePercentage=20):
    _from, _to = int(len(histogram) // (100/ignorePercentage)),\
        int(len(histogram) - len(histogram) // (100//ignorePercentage))

    start, end, step = _from, _to + 1, 1
    if trend == DIRECTION.DOWN:
        start, end, step = _to, _from - 1, -1

    answers = []
    prevBin = 0
    Half = False

    minIdxs = np.where(histogram == np.min(histogram[_from:_to]))
    for minIdx in minIdxs[0]:
        answers.append({'type': "type1", "index": minIdx})
    for i in range(start, end, step):
        curBin = histogram[i]
        if curBin < prevBin / 2:
            Half = True
        elif Half:
            answers.append({"type": "type2", "index": i - 1})
            Half = False
        prevBin = curBin

    return answers


def _getTPs(vpdf, tpsIdx, trend):
    res = set()
    for _, dic in enumerate(tpsIdx):
        price = vpdf.iloc[dic['index']].minPrice if trend == DIRECTION.UP else vpdf.iloc[dic['index']].maxPrice
        res.add(price)   
    return res   

def getTPs(df: pd.DataFrame, tradeSide, nBins=20):
    """suggest target points based on wave
    
    params:
        df: pd.DataFrame -> appropriate for volume profile which I had explained in the volprofile package. Checkout `volprofile.getVP` function.
                            Also it must provide the basic ohlcv data.
        tradeSide: str: ['UP', 'DOWN']
        nBins: int -> needed for volume profile (default: 20)
        
    """

def _test():
    path = "~/Downloads/data/tickers_data/test.csv"
    df = pd.read_csv(path)

    nBins = 20
    basedOnWhichWaveFromLast = 5
    ignorePercentage = 20

    # entrypoint
    # type 2 input
    # type 3 input
    # type 4 : input window , input threshold of size 
    # trade side
    # nth wave of same trade side
    # range

    n= 500
    df = df[-n:]
    pivots = peak_valley_pivots(df.close, 0.3, -0.3)
    trend = findTrend(pivots)
    waveIndices = _findWave(pivots, basedOnWhichWaveFromLast)

    df['price'] = (df['high'] + df['low']) / 2
    df = df[['volume', 'price']]

    forPlot = df[-n:]
    print(forPlot)
    df = df[waveIndices[0] : waveIndices[1]]


    res = vp.getVP(df, nBins=nBins)
    TPsIdx = _getTPsIdx(res.aggregateVolume, trend, ignorePercentage=ignorePercentage)
    TPs = _getTPs(res, TPsIdx, trend)

    fig = make_subplots(rows = 1, cols = 2)
    fig.add_trace(go.Bar(
                x=res.aggregateVolume,
                y=(res.minPrice + res.maxPrice) / 2,
                orientation='h'), row=1, col=2)

    fig.add_trace(go.Scatter(name='close', x=np.arange(len(forPlot.price)), y=forPlot.price,
                mode='lines', marker_color='#D2691E'))
    fig.add_trace(go.Scatter(name='top', x=np.arange(len(forPlot.price))[
        pivots == 1], y=forPlot.price[pivots == 1], mode='markers', marker_color='green'))
    fig.add_trace(go.Scatter(name='top', x=np.arange(len(forPlot.price))[
                pivots == -1], y=forPlot.price[pivots == -1], mode='markers', marker_color='red'))

    for line in TPs:
        fig.add_hline(y=line, line_width=3, line_dash="dash", line_color="green")

    fig.show()

if __name__ == '__main__':
    _test()