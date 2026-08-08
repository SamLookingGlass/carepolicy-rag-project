from src.eval.metrics import _cosine, answer_similarity


def test_cosine():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert _cosine([1.0, 1.0], [0.0, 0.0]) == 0.0  # zero vector guard
    assert abs(_cosine([1.0, 0.0], [1.0, 1.0]) - 0.5**0.5) < 1e-9


def test_answer_similarity_skips_refusals():
    # No reference answer (refusal cases) -> None, no embedding call made
    assert answer_similarity("some answer", "") is None
