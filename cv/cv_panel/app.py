"""Minimal Dash mock-up for experimenting with image transformations."""

from __future__ import annotations
from typing import Optional
from dataclasses import asdict

import dash_mantine_components as dmc
from dash import Dash, Input, Output, html, dcc
from dash_iconify import DashIconify

from models import AppState, Menu
from methods import image_registration

app_state = AppState(
    methods=[image_registration.method_state],
)

app: Dash = Dash(__name__, title="Dash Image Transform Panel")


def make_navbar(app_state: AppState) -> dmc.AppShellNavbar:
    return dmc.AppShellNavbar(
        id="navbar",
        children=[
            dmc.NavLink(
                label=method.name,
                leftSection=DashIconify(icon="hugeicons:apple-vision-pro", height=16),
                variant="subtle",
                active=True,
                p="4",
                c="gray.4",
            )
            for method in app_state.methods
        ],
        p=0,
    )


def make_menu(menu: Menu, depth=1) -> dmc.Menu:
    def get_menu_id(menu_def: Menu) -> str:
        match menu_def.action:
            case str(action_id):
                return action_id
            case [*_]:
                return menu_def.name
            case _:
                raise ValueError(f"Unknown action type: {type(menu_def.action)}")

    def get_items(menu_def: Menu) -> Optional[list[Menu]]:
        match menu_def.action:
            case str():
                return None
            case [*menus] if all(isinstance(v, Menu) for v in menus):
                return menus
            case _:
                raise ValueError(f"Unknown submenu type: {type(menu_def.action)}")

    MenuComponent = dmc.SubMenu if depth > 1 else dmc.Menu
    MenuTargetComponent = dmc.SubMenuTarget if depth > 1 else dmc.MenuTarget
    MenuDropdownComponent = dmc.SubMenuDropdown if depth > 1 else dmc.MenuDropdown
    MenuItemComponent = dmc.SubMenuItem if depth > 1 else dmc.MenuItem

    children = [MenuTargetComponent(children=dmc.Button(children=menu.name, id=get_menu_id(menu)))]
    if sub_items := get_items(menu):
        children.append(
            MenuDropdownComponent(
                children=[
                    MenuItemComponent(subitem.name) if subitem.is_leaf else make_menu(subitem, depth=depth + 1)
                    for subitem in sub_items
                ]
            )
        )
    return MenuComponent(
        trigger="hover",
        openDelay=100,
        closeDelay=400,
        children=children,
        transitionProps={"transition": "rotate-right", "duration": 150},
    )


def make_header(app_state: AppState) -> dmc.AppSheelHeader:
    return dmc.AppShellHeader(
        dmc.Group(
            id="header-group",
            children=[make_menu(menu) for menu in app_state.menu],
            justify="flex-start",
            gap="xs",
            grow=False,
            px="md",
        ),
    )


layout = dmc.AppShell(
    children=[
        dcc.Store(id="app_state", data=asdict(app_state)),
        make_header(app_state),
        make_navbar(app_state),
        dmc.AppShellMain("Aside is hidden on md breakpoint and cannot be opened when it is collapsed"),
        dmc.AppShellAside("Aside", p="md"),
        dmc.AppShellFooter("Footer", px="xl"),
    ],
    header={"height": 32},
    footer={"height": 24},
    navbar={
        "width": 160,
        "breakpoint": "sm",
        "collapsed": {"mobile": True},
    },
    aside={
        "width": 160,
        "breakpoint": "sm",
        "collapsed": {"desktop": False, "mobile": True},
    },
    padding="0",
    id="appshell",
)
app.layout = dmc.MantineProvider(
    children=[layout],
    forceColorScheme="dark",
)


@app.callback(
    Output("app-state", "children"),
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
    app.callback()
