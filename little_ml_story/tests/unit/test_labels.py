from apps.api.app.imagenet_labels import LABELS, label_for


def test_known_label():
    assert label_for(281) == "tabby_cat"


def test_unknown_label_falls_back():
    assert label_for(999) == "class_999"


def test_labels_distinct():
    assert len(set(LABELS.values())) == len(LABELS)
