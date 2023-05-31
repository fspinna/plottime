import awkward as ak


def subsequence_signal_to_multivariate(sy, signal_id, n_signals, sx=None):
    SY = [[[] for _ in range(n_signals)]]
    SY[0][signal_id] = sy
    SY = ak.Array(SY)
    if sx is not None:
        SX = [[[] for _ in range(n_signals)]]
        SX[0][signal_id] = sx
        SX = ak.Array(SX)
    else:
        SX = sx
    return SY, SX
