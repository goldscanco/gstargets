from zigzag import peak_valley_pivots
import pandas as pd
from gstargets.config import DIRECTION, Develop
from gstargets.shared import (
    _get_min_idxs,
    _get_jumped_idxs,
    _get_requested_volprofile_for_waves,
    _set_ignore,
    _prepare_dataframe_volumeprofile,
)


def _get_tp_level_index(
    volprofile_result,
    tradeSide,
    ignorePercentageUp=20,
    ignorePercentageDown=20,
    thresholdType2=0.5,
):
    _prepare_dataframe_volumeprofile(volprofile_result)
    _set_ignore(ignorePercentageUp, ignorePercentageDown, volprofile_result)

    answers = []
    for minIdx in _get_min_idxs(volprofile_result[volprofile_result["valid"]]):
        answers.append({"type": "type1", "index": minIdx})
    for jumpedIdx in _get_jumped_idxs(volprofile_result, thresholdType2, tradeSide):
        answers.append({"type": "type2", "index": jumpedIdx})
    if Develop:
        print(answers)
    return answers


def _convert_index_to_price(vpdf, tpsIdx, trend):
    res = []
    for _, dic in enumerate(tpsIdx):
        price = (
            vpdf.iloc[dic["index"]].minPrice
            if trend == DIRECTION.UP
            else vpdf.iloc[dic["index"]].maxPrice
        )
        res.append({"price": price, "type": dic["type"]})
    if Develop:
        print(res)
    return res


def getTPs(
    df: pd.DataFrame,
    tradeSide,
    thresholdType2=0.5,
    upWaveNums=[],
    downWaveNums=[],
    nBins=20,
    ignorePercentageUp=20,
    ignorePercentageDown=20,
    zigzagUpThreshold=0.3,
    zigzagDownThreshold=-0.3,
    returnVP=False,
    returnPivot=False,
):
    """suggest target points based on wave

    params:
        df: pd.DataFrame -> appropriate for volume profile which I had explained in the volprofile package.
                            Checkout `volprofile.getVP` function.
                        It means that it should have `price` and `volume`
                        Also it must provide the basic ohlcv data.
        tradeSide: str: ['UP', 'DOWN']
        thresholdType2: float -> calculate type2 TPs based on this
        upWaveNums: list[int] -> Exclusive list of the upward waves to be selected for volume profile indicator.
                        It starts from 1 which means the last upward wave except for the current wave.
                        default ([])
        downWaveNums: list[int] -> same as upWaveNums but in downWard direction default ([])
        nBins: int -> needed for volume profile
                        default (20)
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

    pivots = peak_valley_pivots(df.close, zigzagUpThreshold, zigzagDownThreshold)

    res = _get_requested_volprofile_for_waves(
        df, pivots, upWaveNums, downWaveNums, nBins
    )
    TPsIdx = _get_tp_level_index(
        res,
        tradeSide,
        ignorePercentageUp,
        ignorePercentageDown,
        thresholdType2=thresholdType2,
    )
    TPs = _convert_index_to_price(res, TPsIdx, tradeSide)
    toReturn = TPs
    # the following line are only for testing purposes
    if returnVP or returnPivot:
        toReturn = {"tps": TPs}
    if returnVP:
        toReturn["vp"] = res
    if returnPivot:
        toReturn["pivots"] = pivots
    return toReturn
