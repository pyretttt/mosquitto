from __future__ import annotations

import reflex as rx

from reflex_app.models import Menu, Method
from reflex_app.state import APP_STATE, AppViewState


def method_nav_button(method: Method) -> rx.Component:
    """Button that selects a specific method."""
    return rx.button(
        method.name,
        width="100%",
        justify_content="flex-start",
        variant=rx.cond(AppViewState.selected_method_id == method.id, "solid", "ghost"),
        color_scheme=rx.cond(AppViewState.selected_method_id == method.id, "blue", "gray"),
        size="1",
        on_click=AppViewState.select_method(method.id),
    )


def navbar() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.text("Methods", font_weight="bold"),
            *[method_nav_button(method) for method in APP_STATE.methods],
            spacing="2",
            align="stretch",
        ),
        width="220px",
        padding="0rem",
        border_right="1px solid var(--gray-5)",
        bg="var(--gray-1)",
        # min_h="calc(100vh - 64px)",
    )


def menu_dropdown(menu: Menu) -> rx.Component:
    if menu.is_leaf:
        action_id = str(menu.action)
        return rx.button(
            menu.name,
            size="1",
            variant="ghost",
            on_click=AppViewState.trigger_menu_action(action_id),
        )
    return rx.menu.root(
        rx.menu.trigger(
            rx.button(menu.name, size="1", variant="outline"),
        ),
        rx.menu.content(
            *[menu_item(submenu) for submenu in menu_items(menu)],
        ),
    )


def menu_item(menu: Menu) -> rx.Component:
    if menu.is_leaf:
        return rx.menu.item(
            menu.name,
            on_select=AppViewState.trigger_menu_action(str(menu.action)),
        )
    return rx.menu.sub(
        rx.menu.sub_trigger(
            rx.flex(
                rx.text(menu.name),
                rx.text("›", color="gray"),
                gap="0.25rem",
                align="center",
            )
        ),
        rx.menu.sub_content(
            *[menu_item(submenu) for submenu in menu_items(menu)],
        ),
    )


def menu_items(menu: Menu) -> list[Menu]:
    return menu.action if isinstance(menu.action, list) else []


def header() -> rx.Component:
    return rx.flex(
        rx.button(
            rx.cond(AppViewState.navbar_collapsed, "Show menu", "Hide menu"),
            variant="ghost",
            size="1",
            on_click=AppViewState.toggle_navbar,
        ),
        rx.flex(
            *[menu_dropdown(menu) for menu in APP_STATE.menu],
            gap="0.5rem",
        ),
        rx.spacer(),
        rx.text("Reflex CV Panel", font_weight="bold"),
        align="center",
        padding="0.75rem 1rem",
        border_bottom="1px solid var(--gray-5)",
        bg="var(--gray-2)",
        gap="0.75rem",
    )


def option_card(option) -> rx.Component:
    return rx.box(
        rx.text(option["name"], font_weight="medium"),
        rx.text(option["value"], color="gray"),
        rx.cond(
            option["description"] != "",
            rx.text(option["description"], color="gray"),
            rx.box(),
        ),
        border="1px solid var(--gray-5)",
        border_radius="8px",
        padding="0.75rem",
        width="100%",
    )


def method_details() -> rx.Component:
    return rx.vstack(
        rx.heading(
            rx.cond(
                AppViewState.selected_method_name.length() > 0,
                AppViewState.selected_method_name,
                "Select a method",
            ),
            size="4",
        ),
        rx.text(AppViewState.selected_method_description, color="gray"),
        rx.foreach(
            AppViewState.selected_method_options,
            lambda option: option_card(option),
        ),
        spacing="6",
        width="100%",
    )


def main_content() -> rx.Component:
    return rx.box(
        method_details(),
        flex="1",
        padding="1.5rem",
        width="100%",
    )


def footer() -> rx.Component:
    return rx.flex(
        rx.text(AppViewState.last_menu_action, font_size="1", color="gray"),
        justify="between",
        padding="0.5rem 1.5rem",
        border_top="1px solid var(--gray-5)",
        bg="var(--gray-1)",
    )


def app_shell() -> rx.Component:
    return rx.box(
        header(),
        rx.flex(
            rx.cond(
                AppViewState.navbar_collapsed,
                rx.box(width="0px"),
                navbar(),
            ),
            rx.box(
                main_content(),
                footer(),
                display="flex",
                flex_direction="column",
                flex="1",
                min_h="calc(100vh - 64px)",
            ),
            width="100%",
        ),
        display="flex",
        flex_direction="column",
        min_h="100vh",
        bg="var(--gray-2)",
    )


def index() -> rx.Component:
    return app_shell()


app = rx.App()
app.add_page(index, route="/", title="Image Transform Panel")
# app.compile()
