from zigzag import peak_valley_pivots
import pandas as pd
from gstargets.config import DIRECTION, Develop
from gstargets.shared import (
    _get_high_volume_area,
    _get_requested_volprofile_for_waves,
    _set_ignore,
    _findWave,
    _fractional_candle,
    _prepare_dataframe_volumeprofile,
)


def getReversalArea(
    df: pd.DataFrame,
    tradeSide,
    entryPoint=None,
    upWaveNums=[1],
    downWaveNums=[1],
    nBins=20,
    ignorePercentageUp=20,
    ignorePercentageDown=20,
    zigzagUpThreshold=0.3,
    zigzagDownThreshold=-0.3,
    returnVP=False,
    returnPivot=False,
):
    reversalAreas = []
    pivots = peak_valley_pivots(df.close, zigzagUpThreshold, zigzagDownThreshold)
    volprofile_result = _get_requested_volprofile_for_waves(
        df, pivots, upWaveNums, downWaveNums, nBins
    )
    _prepare_dataframe_volumeprofile(volprofile_result)
    _set_ignore(ignorePercentageUp, ignorePercentageDown, volprofile_result)

    res = _get_high_volume_area(
        volprofile_result,
        current_price=entryPoint,
        trade_side=tradeSide,
    )
    reversalAreas.extend(res)
    # the following line are only for testing purposes
    if Develop:
        print(reversalAreas)
    _expand_reversal_areas(df, reversalAreas, tradeSide, pivots, zigzagUpThreshold, zigzagDownThreshold)
    if Develop:
        print(reversalAreas)
    toReturn = reversalAreas
    if returnVP or returnPivot:
        toReturn = {"reversal_areas": toReturn}
    if returnVP:
        toReturn["vp"] = volprofile_result
    if returnPivot:
        toReturn["pivots"] = pivots
    return toReturn


def _expand_reversal_areas(df, reversalAreas, tradeSide, pivots, upThr, downThr):
    expand = "maxPrice"
    if tradeSide == DIRECTION.UP:
        expand = "minPrice"

    after, before = _findWave(pivots, waveNum=1, direction=DIRECTION.UP if tradeSide == DIRECTION.DOWN else DIRECTION.DOWN)

    print(after, before)
    # first iteration to get the approximate area
    for _, row in df.iloc[after:before, :].iterrows():
        for j, rev_area in enumerate(reversalAreas):
            _dict = row.to_dict()
            if (
                _fractional_candle(0.5, _dict)
                and _dict["high"] > rev_area[expand]
                and _dict["low"] < rev_area[expand]
            ):
                reversalAreas[j][expand] = (
                    _dict["high"] if tradeSide == DIRECTION.DOWN else _dict["low"]
                )
    # TODO remove hard-code
    micro_pivots = peak_valley_pivots(df.close, upThr/5, downThr/5)

    valid_pivots = set()
    Idx = 1
    while True:
        leftU, _ = _findWave(micro_pivots, Idx, direction=DIRECTION.UP)
        leftD, _ = _findWave(micro_pivots, Idx, direction=DIRECTION.DOWN)

        if leftU < before and leftU > after:
            valid_pivots.add(leftU)
        if leftD < before and leftD > after:
            valid_pivots.add(leftD)

        if leftD < after and leftU < after:
            break
        Idx += 1

    for idx, row in df.iloc[after:before, :].iterrows():
        for j, rev_area in enumerate(reversalAreas):
            _dict = row.to_dict()
            if (
                abs(_dict['close'] - rev_area[expand]) / rev_area[expand] < 0.13
                and idx in valid_pivots
            ):
                reversalAreas[j][expand] = (
                    _dict["high"] if tradeSide == DIRECTION.DOWN else _dict["low"]
                )
