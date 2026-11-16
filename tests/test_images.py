from scripts.fetch_images import WIKI_TITLES, clean_html, parse_image_info, parse_page_image
from data.validate import load_cars


def test_every_model_has_a_wikipedia_title():
    models = set(load_cars()[["brand", "model"]].drop_duplicates().itertuples(index=False, name=None))
    assert models == set(WIKI_TITLES)


def test_parse_page_image():
    api = {"query": {"pages": {"1": {"title": "Toyota Corolla", "pageimage": "Corolla_E210.jpg",
                                     "thumbnail": {"source": "https://upload.wikimedia.org/x/800px-Corolla_E210.jpg"}}}}}
    assert parse_page_image(api) == ("Corolla_E210.jpg", "https://upload.wikimedia.org/x/800px-Corolla_E210.jpg")


def test_parse_page_image_missing():
    assert parse_page_image({"query": {"pages": {"-1": {"title": "Nope", "missing": ""}}}}) is None


def test_parse_image_info_free_license():
    api = {"query": {"pages": {"5": {"imageinfo": [{"descriptionurl": "https://commons.wikimedia.org/wiki/File:X.jpg",
        "extmetadata": {"LicenseShortName": {"value": "CC BY-SA 4.0"},
                        "Artist": {"value": '<a href="//commons.wikimedia.org/wiki/User:Ali">Ali Khan</a>'}}}]}}}}
    info = parse_image_info(api)
    assert info == {"file_page": "https://commons.wikimedia.org/wiki/File:X.jpg", "license": "CC BY-SA 4.0", "artist": "Ali Khan"}


def test_parse_image_info_rejects_non_free():
    api = {"query": {"pages": {"5": {"imageinfo": [{"descriptionurl": "u", "extmetadata": {
        "LicenseShortName": {"value": "Fair use"}, "Artist": {"value": "x"}}}]}}}}
    assert parse_image_info(api) is None


def test_clean_html():
    assert clean_html("<span>Mr&nbsp;<b>X</b></span>") == "Mr X"


def test_file_matches_model():
    from scripts.fetch_images import file_matches_model
    assert file_matches_model("2019_Honda_BR-V_E_CVT.jpg", "BR-V")
    assert not file_matches_model("Six_cars_in_the_NMNH.jpg", "BR-V")
    assert file_matches_model("Toyota_Yaris_ATIV_2018.jpg", "Yaris")
    assert file_matches_model("OSAKA_-_Navic_ALTO_HA36.JPG", "Alto")


def test_file_matches_multiword_model():
    from scripts.fetch_images import file_matches_model
    assert file_matches_model("2022_Kia_Sportage_NQ5_front.jpg", "Sportage L")
    assert file_matches_model("Changan_Oshan_X7_2022.jpg", "Oshan X7")
    assert not file_matches_model("Changan_Oshan_X5.jpg", "Oshan X7")
