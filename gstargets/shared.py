import pandas as pd
from zigzag import peak_valley_pivots
import numpy as np
import volprofile as vp
from gstargets.config import DIRECTION, Develop


def _fractional_candle(
    fraction=0.5, _dict={"open": None, "high": None, "low": None, "close": None}
):
    shadow = _dict["high"] - _dict["low"]
    body = abs(_dict["open"] - _dict["close"])

    return body / shadow < fraction


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


def _get_min_idxs(volprofile_result):
    idxs = np.where(
        volprofile_result.aggregateVolume == np.min(volprofile_result.aggregateVolume)
    )[0]

    answers = []
    for idx in idxs:
        answers.append(int(volprofile_result.iloc[idx, :]["index"]))
    return answers


def _get_max_idxs(volprofile_result):
    idxs = np.where(
        volprofile_result.aggregateVolume == np.max(volprofile_result.aggregateVolume)
    )[0]

    answers = []
    for idx in idxs:
        answers.append(int(volprofile_result.iloc[idx, :]["index"]))
    return answers


def _get_jumped_idxs(volprofile_result: pd.DataFrame, thresholdType2, tradeSide):
    step = 1 if tradeSide == DIRECTION.UP else -1
    answers = []
    Half = False
    prevBin = {"aggregateVolume": 0}
    for _, row in volprofile_result[::step].iterrows():
        if not row["valid"]:
            continue
        if row["aggregateVolume"] < prevBin["aggregateVolume"] * thresholdType2:
            Half = True
        elif Half:
            answers.append(prevBin["index"])
            Half = False
        prevBin = row

    return answers


def _get_high_volume_area(
    volprofile_result: pd.DataFrame,
    current_price,
    trade_side,
):
    if trade_side == DIRECTION.DOWN:
        volprofile_result[volprofile_result.minPrice > current_price]["valid"] &= True
    else:
        volprofile_result[volprofile_result.maxPrice < current_price]["valid"] &= True

    _maxes = _get_max_idxs(volprofile_result[volprofile_result["valid"]])
    return [volprofile_result.iloc[_max, :].to_dict() for _max in _maxes]


def _prepare_dataframe_volumeprofile(df: pd.DataFrame):
    df["index"] = df.index
    df["valid"] = False


def _set_ignore(upP, downP, df: pd.DataFrame):
    _len = len(df)
    start, end = (
        int(_len / 100 * downP),
        int(_len - _len / 100 * upP),
    )
    df.loc[start:end, "valid"] = True


def _get_requested_volprofile_for_waves(df, pivots, upWaveNums, downWaveNums, nBins):
    df["inVolumeProfile"] = False

    for upWaveNum in upWaveNums:
        waveIndices = _findWave(pivots, upWaveNum, DIRECTION.UP)
        cond1 = np.logical_and(df.index >= waveIndices[0], df.index < waveIndices[1])
        df["inVolumeProfile"] = np.logical_or(cond1, df["inVolumeProfile"])

    for downWaveNum in downWaveNums:
        waveIndices = _findWave(pivots, downWaveNum, DIRECTION.DOWN)
        cond1 = np.logical_and(df.index >= waveIndices[0], df.index < waveIndices[1])
        df["inVolumeProfile"] = np.logical_or(cond1, df["inVolumeProfile"])

    df = df[df.inVolumeProfile == True]

    res = vp.getVP(df, nBins=nBins)
    res["index"] = res.index
    return res
