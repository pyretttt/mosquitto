"""Minimal Dash mock-up for experimenting with image transformations."""

from __future__ import annotations
from typing import List
import os
import uuid

import yaml
import dash_mantine_components as dmc
from dash import Dash, Input, Output, html

from models import Method, Option, AppState, OptionVariant, NumberField, ValueSelector, Field, Checkbox


def make_option_info(option: dict) -> OptionVariant:
    option_type = option.get("type")
    if option_type == "number_field":
        return NumberField(
            value=option.get("value", float("nan")),
            min_value=option.get("min_value", float("nan")),
            max_value=option.get("max_value", float("nan")),
        )
    elif option_type == "value_selector":
        values = option.get("values", [])
        idx = option.get("selected_idx", 0)
        assert len(values) and idx < len(values) and idx >= 0
        return ValueSelector(
            values=values,
            selected_idx=idx,
        )
    elif option_type == "field":
        return Field(value=option.get("value", ""))
    elif option_type == "checkbox":
        return Checkbox(value=option.get("value", False))


def load_app_state(descriptors_dir: str) -> AppState:
    methods: List[Method] = []
    if not os.path.isdir(descriptors_dir):
        # return make_debug_app_state()
        pass

    for root, _, files in os.walk(descriptors_dir):
        for file in files:
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                info = config.get("info") or {}
                id = str(uuid.uuid4())

                options = [
                    Option(
                        id=raw_option.get("id"),
                        name=raw_option.get("name", "L10n.name"),
                        description=raw_option.get("description"),
                        info=make_option_info(option=raw_option),
                    )
                    for raw_option in config.get("options", [])
                ]
                methods.append(
                    Method(
                        id=id,
                        title=info.get("name", "L10n.title"),
                        description=info.get("description", "L10n.description"),
                        options=options,
                    )
                )

            except Exception as e:
                print("Failed to parse config at:", path, "error:", repr(e))

    return AppState(methods=methods, selected_id=None)


app_state = load_app_state("descriptors")
print(app_state)

TRANSFORMATIONS = [
    {"name": "Grayscale"},
    {"name": "Blur"},
    {"name": "Edge Detection"},
    {"name": "Perspective Warp"},
    {"name": "Histogram Equalization"},
]

TRANSFORMATION_OPTIONS = {
    "Grayscale": ["Channel averaging", "Keep alpha"],
    "Blur": ["Kernel size: 5x5", "Gaussian"],
    "Edge Detection": ["Canny", "Threshold 1 / Threshold 2"],
    "Perspective Warp": ["4-point homography", "Bicubic resampling"],
    "Histogram Equalization": ["CLAHE", "Tile size 8x8"],
}

app: Dash = Dash(__name__, title="Dash Image Transform Panel")


layout = dmc.AppShell(
    [
        dmc.AppShellHeader(
            dmc.Group(
                [
                    dmc.Burger(
                        id="burger",
                        size="sm",
                        hiddenFrom="sm",
                        opened=False,
                    ),
                    dmc.Title("Demo App", c="blue"),
                ],
                h="100%",
                px="md",
            )
        ),
        dmc.AppShellNavbar(
            id="navbar",
            children=[
                "Navbar",
                *[dmc.Skeleton(height=28, mt="sm", animate=False) for _ in range(15)],
            ],
            p="md",
        ),
        dmc.AppShellMain("Aside is hidden on md breakpoint and cannot be opened when it is collapsed"),
        dmc.AppShellAside("Aside", p="md"),
        dmc.AppShellFooter("Footer", p="md"),
    ],
    header={"height": 60},
    footer={"height": 60},
    navbar={
        "width": 300,
        "breakpoint": "sm",
        "collapsed": {"mobile": True},
    },
    aside={
        "width": 300,
        "breakpoint": "md",
        "collapsed": {"desktop": False, "mobile": True},
    },
    padding="md",
    id="appshell",
)
app.layout = dmc.MantineProvider(layout)


@app.callback(
    Output("selected-transform", "children"),
    Output("option-list", "children"),
    Input("transform-table", "selected_rows"),
)
def update_selection(selected_rows: list[int] | None):
    row_index = selected_rows[0] if selected_rows else 0
    row_index = max(0, min(row_index, len(TRANSFORMATIONS) - 1))
    name = TRANSFORMATIONS[row_index]["name"]
    options = TRANSFORMATION_OPTIONS.get(name, ["No options available."])
    return name, [html.Li(option) for option in options]


if __name__ == "__main__":
    app.run(debug=True)
