import jiwer

_NORMALIZE = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def compute_wer(reference: str, hypothesis: str) -> float:
    return jiwer.wer(reference, hypothesis, reference_transform=_NORMALIZE, hypothesis_transform=_NORMALIZE)
