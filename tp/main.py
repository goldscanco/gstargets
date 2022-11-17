from zigzag import *
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

path = "~/Downloads/data/tickers_data/test.csv"
df = pd.read_csv(path)
df = df[-500:]
close = df.close
pivots = peak_valley_pivots(close, 0.3, -0.3)
UP = 'up'
DOWN = 'down'


def findTrend(pivots):
    return UP if pivots[pivots != 0][-2] == -1 else DOWN


def findWave(pivots, waveNum=1):
    indices = np.where(pivots != 0)[0]
    return indices[-waveNum - 1], indices[-waveNum]


def getDevelopmentHistogram(rawHist):
    volume, price, nbin = rawHist['x'], rawHist['y'], rawHist['nbinsy']
    minPrice, maxPrice = np.min(price), np.max(price)

    step = (maxPrice - minPrice) / nbin
    idxs = (price - minPrice) // step

    idxs[idxs >= nbin] = nbin - 1
    idxs[idxs < 0] = 0

    volumes = np.zeros(shape=[nbin])
    prices = np.zeros(shape=[nbin])
    for i, key in enumerate(idxs):
        volumes[int(key)] += volume[i]
    for i in range(nbin):
        prices[i] = getPrice(i, step, minPrice)

    return prices, volumes, step, minPrice


def getPrice(idx, step, minPrice, lowerBound=False, upperBound=False):
    if not lowerBound and not upperBound:
        return minPrice + (2 * idx + 1) * step / 2
    if lowerBound:
        return minPrice + idx * step
    if upperBound:
        return minPrice + (idx + 1) * step


def getTPsIdx(histogram, trend, ignorePercentage=20):
    _from, _to = int(len(histogram) // (100/ignorePercentage)),\
        int(len(histogram) - len(histogram) // (100//ignorePercentage))

    start, end, step = _from, _to + 1, 1
    if trend == DOWN:
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


def getTPs(tpsIdx, trend, step, minPrice):
    lowerBound, upperBound = False, False
    if trend == UP:
        lowerBound = True
    elif trend == DOWN:
        upperBound = True

    res = set()
    for _, dic in enumerate(tpsIdx):
        price = getPrice(dic['index'], step, minPrice, lowerBound, upperBound)
        res.add(price)   
    return res   

nbin = 20
basedOnWhichWaveFromLast = 2
ignorePercentage = 20

# entrypoint
# type 2 input
# type 3 input
# type 4 : input window , input threshold of size 
# trade side
# nth wave of same trade side
# range

trend = findTrend(pivots)
waveIndices = findWave(pivots, basedOnWhichWaveFromLast)

volume = df['volume'].iloc[waveIndices[0]:waveIndices[1]]
price = (df['high'].iloc[waveIndices[0]:waveIndices[1]] +
         df['low'].iloc[waveIndices[0]:waveIndices[1]]) / 2


hist = go.Histogram(x=volume,
                    y=price,
                    nbinsy=nbin,
                    orientation='h'
                    )

histPrice, histVol, step, minP = getDevelopmentHistogram(hist)
TPsIdx = getTPsIdx(histVol, trend, ignorePercentage=ignorePercentage)
TPs = getTPs(TPsIdx, trend, step, minP)



fig = make_subplots(rows = 1, cols = 2)
fig.add_trace(go.Bar(
            x=histVol,
            y=histPrice,
            orientation='h'), row=1, col=2)

fig.add_trace(go.Scatter(name='close', y=df.close,
              mode='lines', marker_color='#D2691E'))
fig.add_trace(go.Scatter(name='top', x=np.arange(len(close))[
    pivots == 1], y=close[pivots == 1], mode='markers', marker_color='green'))
fig.add_trace(go.Scatter(name='top', x=np.arange(len(close))[
              pivots == -1], y=close[pivots == -1], mode='markers', marker_color='red'))

for line in TPs:
    fig.add_hline(y=line, line_width=3, line_dash="dash", line_color="green")

fig.show()
