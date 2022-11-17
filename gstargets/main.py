from zigzag import *
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import volprofile as vp
from gstargets.config import DIRECTION


def _findWave(pivots, waveNum, direction):
    indices = np.where(pivots != 0)[0]
    count, idx = 0, len(indices) - 2
    while count < waveNum and idx != 0:
        _index = indices[idx]
        _currentDir = DIRECTION.UP if pivots[_index] == 1 else DIRECTION.DOWN
        if direction == _currentDir:
            count += 1
        idx -= 1
    return indices[idx], indices[idx + 1]


def _getTPsIdx(histogram, tradeSide, ignorePercentageUp=20, ignorePercentageDown=20, thresholdType2=0.5):
    _from, _to = int(len(histogram) // (100/ignorePercentageDown)),\
        int(len(histogram) - len(histogram) // (100//ignorePercentageUp))

    start, end, step = _from, _to + 1, 1
    if tradeSide == DIRECTION.DOWN:
        start, end, step = _to, _from - 1, -1

    answers = []
    prevBin = 0
    Half = False

    minIdxs = np.add(
        np.where(histogram[_from:_to] == np.min(histogram[_from:_to])), _from)
    for minIdx in minIdxs[0]:
        answers.append({'type': "type1", "index": minIdx})
    for i in range(start, end, step):
        curBin = histogram[i]
        if curBin < prevBin * thresholdType2:
            Half = True
        elif Half:
            answers.append({"type": "type2", "index": i - 1})
            Half = False
        prevBin = curBin

    return answers


def _getTPs(vpdf, tpsIdx, trend):
    res = []
    for _, dic in enumerate(tpsIdx):
        price = vpdf.iloc[dic['index']
                          ].minPrice if trend == DIRECTION.UP else vpdf.iloc[dic['index']].maxPrice
        res.append({"price": price, "type": dic['type']})
    return res


def getTPs(df: pd.DataFrame, tradeSide, maximumAcceptableBarType3=0,
           thresholdType2=0.5, entryPoint=None, upWaveNums=[], downWaveNums=[],
           nBins=20, windowType3=2, ignorePercentageUp=20, ignorePercentageDown=20,
           zigzagUpThreshold=0.3, zigzagDownThreshold=-0.3, returnVP=False, returnPivot=False):
    """suggest target points based on wave

    params:
        df: pd.DataFrame -> appropriate for volume profile which I had explained in the volprofile package. Checkout `volprofile.getVP` function.
                        It means that it should have `price` and `volume`
                        Also it must provide the basic ohlcv data.
        tradeSide: str: ['UP', 'DOWN']
        maximumAcceptableBarType3: int -> calculate type3 TPs based on this
        thresholdType2: float -> calculate type2 TPs based on this
        entryPoint: float -> 
                        default (None)
        upWaveNums: list[int] -> Exclusive list of the upward waves to be selected for volume profile indicator.
                        It starts from 1 which means the last upward wave except for the current wave.
                        default ([])
        downWaveNums: list[int] -> same as upWaveNums but in downWard direction default ([])
        nBins: int -> needed for volume profile 
                        default (20)    
        windowType3: int -> calculate type3 TPs based on this
                        default (nBins // 10)
        ignorePercentageUp: int -> ignore the results of the volume profile calculation from the top
                        default (20)
        ignorePercentageDown: int -> ignore the results of the volume profile calculation from the bottom
                        default (20)
        zigzagUpThreshold: float -> default 0.3 
        zigzagDownThreshold: float -> default 0.3 
    return:
        list[Dict{'type', 'price'}] 
        types can be either 
            type1 : minimum volume bar
            type2 : devided by half from one to another
            type3 : before some strong volume bars
    """

    df.reset_index(inplace=True, drop=True)

    pivots = peak_valley_pivots(
        df.close, zigzagUpThreshold, zigzagDownThreshold)

    df['inVolumeProfile'] = False

    for upWaveNum in upWaveNums:
        waveIndices = _findWave(pivots, upWaveNum, DIRECTION.DOWN)
        cond1 = np.logical_and(
            df.index >= waveIndices[0], df.index < waveIndices[1])
        df['inVolumeProfile'] = np.logical_or(cond1, df['inVolumeProfile'])

    for downWaveNum in downWaveNums:
        waveIndices = _findWave(pivots, downWaveNum, DIRECTION.DOWN)
        cond1 = np.logical_and(
            df.index >= waveIndices[0], df.index < waveIndices[1])
        df['inVolumeProfile'] = np.logical_or(cond1, df['inVolumeProfile'])

    df = df[df.inVolumeProfile == True]

    res = vp.getVP(df, nBins=nBins)
    TPsIdx = _getTPsIdx(res.aggregateVolume, tradeSide,
                        ignorePercentageUp, ignorePercentageDown, thresholdType2=thresholdType2)
    TPs = _getTPs(res, TPsIdx, tradeSide)
    toReturn = TPs
    if returnVP or returnPivot:
        toReturn = {'tps': TPs}
    if returnVP:
        toReturn['vp'] = res
    if returnPivot:
        toReturn['pivots'] = pivots
    return toReturn

def _test():
    path = "~/Downloads/data/tickers_data/test.csv"
    df = pd.read_csv(path)

    nBins = 20
    tradeSide = DIRECTION.UP
    downWaveNums = [1]
    upWaveNums = []
    zigzagDownThreshold = -0.3
    zigzagUpThreshold = 0.3
    ignorePercentageUp = 20
    ignorePercentageDown = 20

    # TODO: type 3 input
    # TODO: increasing type3

    n = 500
    df = df[-n:]
    
    df['price'] = (df['high'] + df['low']) / 2
    df = df[['volume', 'price', 'close']]
    plot(df, tradeSide, nBins=nBins, downWaveNums=downWaveNums)
    
def plot(df: pd.DataFrame, tradeSide, maximumAcceptableBarType3=0,
           thresholdType2=0.5, entryPoint=None, upWaveNums=[], downWaveNums=[],
           nBins=20, windowType3=2, ignorePercentageUp=20, ignorePercentageDown=20,
           zigzagUpThreshold=0.3, zigzagDownThreshold=-0.3):
    """completely identical inputs to getTPs function
        you can use this function to display the results.
    """
    df.reset_index(inplace=True, drop=True)
    forPlot = df.copy()
    # print(df)
    res = getTPs(df, tradeSide, maximumAcceptableBarType3=maximumAcceptableBarType3,
                 thresholdType2=thresholdType2, entryPoint=entryPoint, upWaveNums=upWaveNums,
                 downWaveNums=downWaveNums, nBins=nBins, windowType3=windowType3, ignorePercentageUp=ignorePercentageUp,
                 ignorePercentageDown=ignorePercentageDown, zigzagUpThreshold=zigzagUpThreshold, 
                 zigzagDownThreshold=zigzagDownThreshold, returnPivot=True, returnVP=True)
    TPs, res, pivots = res['tps'], res['vp'], res['pivots']  

    fig = make_subplots(rows = 1, cols = 2)
    fig.add_trace(go.Bar(
                x=res.aggregateVolume,
                y=(res.minPrice + res.maxPrice) / 2,
                orientation='h'), row=1, col=2)

    close = forPlot.close
    fig.add_trace(go.Scatter(name='close', y=close,
                mode='lines', marker_color='#D2691E'))
    fig.add_trace(go.Scatter(name='top', x=np.arange(len(close))[
        pivots == 1], y=close[pivots == 1], mode='markers', marker_color='green'))
    fig.add_trace(go.Scatter(name='top', x=np.arange(len(close))[
                pivots == -1], y=close[pivots == -1], mode='markers', marker_color='red'))

    for line in TPs:
        fig.add_hline(y=line['price'], line_width=3, line_dash="dash", line_color="green")
        
    fig.show()

if __name__ == '__main__':
    _test()
