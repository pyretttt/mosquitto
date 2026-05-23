"""Tiny subset of ImageNet class labels.

We only ship a handful so the repo stays light; the rest fall back to
"class_<id>". Replace with the full 1000-class file in `infra/triton/`
under the advanced track if you need accurate labels.
"""

LABELS: dict[int, str] = {
    0: "tench",
    1: "goldfish",
    207: "golden_retriever",
    208: "Labrador_retriever",
    281: "tabby_cat",
    282: "tiger_cat",
    283: "Persian_cat",
    285: "Egyptian_cat",
    340: "zebra",
    386: "African_elephant",
    817: "sports_car",
    920: "traffic_light",
}


def label_for(class_id: int) -> str:
    return LABELS.get(class_id, f"class_{class_id}")
