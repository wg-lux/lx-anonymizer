import spacy

from lx_anonymizer.ner.spacy_extractor import SpacyModelManager


def test_spacy_model_manager_falls_back_without_download(monkeypatch):
    SpacyModelManager._instance = None
    monkeypatch.delenv(SpacyModelManager.ALLOW_DOWNLOAD_ENV, raising=False)

    def fake_load(_name: str):
        raise OSError("missing model")

    def fake_download(_name: str):
        raise AssertionError("download should not be attempted")

    monkeypatch.setattr(spacy, "load", fake_load)
    monkeypatch.setattr(spacy.cli, "download", fake_download)

    model = SpacyModelManager.get_model("de_core_news_lg")
    assert model.lang == "de"

    SpacyModelManager._instance = None
