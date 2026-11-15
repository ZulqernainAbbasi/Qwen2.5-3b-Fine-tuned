import config


def test_paths_under_root():
    for p in [config.DATA_DIR, config.CARS_CSV, config.DOCS_DIR, config.CHROMA_DIR, config.DB_PATH]:
        assert config.ROOT in p.parents
    assert config.REFUSAL == "I don't have verified information on that."
    assert config.EMBED_MODEL == "BAAI/bge-small-en-v1.5"
