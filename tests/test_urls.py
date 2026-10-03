from aam.ingest.urls import parse_urls, url_key


def test_parse_urls_adds_scheme_dedupes_and_drops_garbage():
    text = "https://a.de/x\n a.de/x/ \nwww.b.de/y#frag,\n\nkeineurl\nhttp://c.de/z?q=1"
    assert parse_urls(text) == ["https://a.de/x", "https://www.b.de/y", "http://c.de/z?q=1"]


def test_parse_urls_empty():
    assert parse_urls("") == []
    assert parse_urls(None) == []


def test_url_key_ignores_scheme_case_and_trailing_slash():
    assert url_key("HTTPS://A.de/x/") == url_key("http://a.de/x") == "a.de/x"
    assert url_key("https://a.de/x?p=1") == "a.de/x?p=1"
    assert url_key("https://a.de/") == "a.de"


def test_url_key_treats_input_without_scheme_like_https():
    assert url_key("A.de/x/") == url_key("https://a.de/x") == "a.de/x"


def test_url_key_never_raises_on_unsplittable_input():
    assert url_key("  HTTP://[ ") == "http://["


def test_parse_urls_skips_malformed_tokens():
    assert parse_urls("http://[ \nhttps://a.de/x") == ["https://a.de/x"]


def test_parse_urls_strips_wrappers_and_drops_mail_addresses():
    assert parse_urls("<https://a.de/x>, (b.de/y). mailto:x@a.de") == ["https://a.de/x", "https://b.de/y"]
    assert parse_urls("\u201ea.de/q\u201c 'c.de/r'") == ["https://a.de/q", "https://c.de/r"]


def test_parse_urls_keeps_brackets_that_belong_to_the_url():
    wiki = "https://de.wikipedia.org/wiki/Dach_(Bauwerk)"
    assert parse_urls(wiki) == [wiki]
    assert parse_urls(f"({wiki})") == [wiki]
    assert parse_urls(f"({wiki}).") == [wiki]
    assert parse_urls("(b.de/y).") == ["https://b.de/y"]
    assert parse_urls("(https://b.de/y)") == ["https://b.de/y"]


def test_parse_urls_keeps_at_signs_in_urls_but_drops_mail_addresses():
    assert parse_urls("https://medium.com/@user/post") == ["https://medium.com/@user/post"]
    assert parse_urls("medium.com/@user") == ["https://medium.com/@user"]
    assert parse_urls("MAILTO:x@a.de x@a.de") == []
