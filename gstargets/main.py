from zigzag import *
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import volprofile as vp
from gstargets.config import DIRECTION


def _findWave(pivots, waveNum, direction):
    """
    get the wave relative to the end of chart for example if waveNum = 1 direction='up'
    get the first wave which the direction was upward
    it follows the notation of zigzag indicator which always ends with a dummy pivot
    """
    # validation check
    if waveNum < 1 or type(waveNum) != int:
        raise ValueError("not a valid waive number")
    if direction not in (DIRECTION.UP, DIRECTION.DOWN):
        raise ValueError("not a valid direction")

    # all trend changing indices
    indices = np.where(pivots != 0)[0]
    # iterate over the first to just one before the last one
    count, idx = 0, len(indices) - 2
    while count < waveNum and idx != 0:
        _index = indices[idx]
        _currentDir = DIRECTION.UP if pivots[_index] == 1 else DIRECTION.DOWN
        if direction == _currentDir:
            count += 1
        idx -= 1
    return indices[idx], indices[idx + 1]

def _get_min_idxs(volprofile_result, start, end):
    return np.add(np.where(volprofile_result[start:end] == np.min(volprofile_result[start:end])), start)[0]

def _get_max_idxs(volprofile_result, start, end):
    return np.add(np.where(volprofile_result[start:end] == np.max(volprofile_result[start:end])), start)[0]

def _get_jumped_idxs(volprofile_result, start, end, step, thresholdType2):
    answers = []
    Half = False
    prevBin = 0
    for i in range(start, end, step):
        curBin = volprofile_result[i]
        if curBin < prevBin * thresholdType2:
            Half = True
        elif Half:
            answers.append(i - 1)
            Half = False
        prevBin = curBin

    return answers

def _get_tp_level_index(volprofile_result, tradeSide,
               ignorePercentageUp=20, ignorePercentageDown=20,
               thresholdType2=0.5):
    volprofile_boxes = len(volprofile_result)
    start, end, step = int(volprofile_boxes / 100 * ignorePercentageDown),\
        int(volprofile_boxes - volprofile_boxes / 100 * ignorePercentageUp), 1

    if tradeSide == DIRECTION.DOWN:
        start, end, step = end, start - 1, -1

    answers = []

    minIdxs = _get_min_idxs(volprofile_result, start, end)
    for minIdx in minIdxs:
        answers.append({'type': "type1", "index": minIdx})

    for jumpedIdx in _get_jumped_idxs(volprofile_result, start, end, step, thresholdType2):
        answers.append({'type': "type2", "index": jumpedIdx})
    
    return answers


def _convert_index_to_price(vpdf, tpsIdx, trend):
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
        df: pd.DataFrame -> appropriate for volume profile which I had explained in the volprofile package. 
                            Checkout `volprofile.getVP` function.
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
            type1 : the level with minimum volume 
            type2 : the level which is significantly jumped by thresholdType2(ex. 0.5) from last bar  
            type3 : before some strong volume level(s) 
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
    TPsIdx = _get_tp_level_index(res.aggregateVolume, tradeSide,
                        ignorePercentageUp, ignorePercentageDown, thresholdType2=thresholdType2)
    TPs = _convert_index_to_price(res, TPsIdx, tradeSide)
    toReturn = TPs
    # the following line are only for testing purposes
    if returnVP or returnPivot:
        toReturn = {'tps': TPs}
    if returnVP:
        toReturn['vp'] = res
    if returnPivot:
        toReturn['pivots'] = pivots
    return toReturn

   
def plot(df: pd.DataFrame, tradeSide, maximumAcceptableBarType3=0,
           thresholdType2=0.5, entryPoint=None, upWaveNums=[], downWaveNums=[],
           nBins=20, windowType3=2, ignorePercentageUp=20, ignorePercentageDown=20,
           zigzagUpThreshold=0.3, zigzagDownThreshold=-0.3):
    """completely identical inputs to getTPs function
        you can use this function to display the results.
    """
    df.reset_index(inplace=True, drop=True)
    forPlot = df.copy()
    
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
    for waveNum in upWaveNums:
        wave = _findWave(pivots, waveNum, DIRECTION.UP)
        fig.add_trace(go.Scatter(name='close', x=np.arange(len(close))[wave[0]: wave[1]], 
                y=close[wave[0]: wave[1]], mode='lines', marker_color='black'))
    for waveNum in downWaveNums:
        wave = _findWave(pivots, waveNum, DIRECTION.DOWN)
        fig.add_trace(go.Scatter(name='close', x=np.arange(len(close))[wave[0]: wave[1]], 
                y=close[wave[0]: wave[1]], mode='lines', marker_color='black'))
    fig.add_trace(go.Scatter(name='top', x=np.arange(len(close))[
        pivots == 1], y=close[pivots == 1], mode='markers', marker_color='green'))
    fig.add_trace(go.Scatter(name='bottom', x=np.arange(len(close))[
                pivots == -1], y=close[pivots == -1], mode='markers', marker_color='red'))

    for line in TPs:
        fig.add_hline(y=line['price'], line_width=3, line_dash="dash", line_color="green")
        
    fig.show()

def _manual_test():
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
    
    df['price'] = (df['high'] + df['low']) / 2
    df = df[['volume', 'price', 'close']]
    plot(df, tradeSide, 
         nBins=nBins, downWaveNums=downWaveNums,
         zigzagUpThreshold=zigzagUpThreshold,
         zigzagDownThreshold=zigzagDownThreshold
         )
 

if __name__ == '__main__':
    _manual_test()
