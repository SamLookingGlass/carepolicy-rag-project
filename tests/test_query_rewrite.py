
def test_rewrite_expands_chas():
    from src.retrieval.query_rewrite import rewrite_query

    result = rewrite_query("What is CHAS eligibility?")
    assert "Community Health Assist Scheme" in result
