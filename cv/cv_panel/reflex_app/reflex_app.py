from __future__ import annotations

import reflex as rx

from reflex_app.state import Menu, Method, AppState


def method_nav_button(method: Method) -> rx.Component:
    """Button that selects a specific method."""
    return rx.button(
        method.name,
        width="100%",
        justify_content="flex-start",
        variant=rx.cond(AppState.selected_method_id == method.id, "solid", "ghost"),
        color_scheme=rx.cond(AppState.selected_method_id == method.id, "blue", "gray"),
        size="1",
        on_click=AppState.select_method(method.id),
    )


def navbar() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.text("Methods", font_weight="bold"),
            rx.foreach(AppState.methods, method_nav_button),
            spacing="4",
            align="stretch",
        ),
        width="160px",
        padding="0.5rem",
        border_right="1px solid var(--gray-5)",
        bg="var(--gray-3)",
        height="100%",
        overflow_y="auto",
        flex_shrink="1",
    )


def menu_dropdown(menu: Menu, depth=1) -> rx.Component:
    return rx.cond(
        menu.is_leaf,
        rx.button(
            menu.name,
            size="1",
            variant="ghost",
            on_click=AppState.trigger_menu_action(str(menu.action_id)),
        ),
        rx.menu.root(
            rx.menu.trigger(
                rx.button(menu.name, size="1", variant="ghost"),
            ),
            rx.menu.content(rx.foreach(menu.submenus, menu_item)),
        ),
    )


def menu_item(menu: Menu) -> rx.Component:
    return rx.menu.item(
        menu.name,
        on_select=AppState.trigger_menu_action(str(menu.action_id)),
    )


def header() -> rx.Component:
    return rx.flex(
        rx.button(
            rx.cond(AppState.navbar_collapsed, "Show menu", "Hide menu"),
            variant="ghost",
            size="1",
            on_click=AppState.toggle_navbar,
        ),
        rx.flex(rx.foreach(AppState.menu, menu_dropdown), gap="0.5rem", align="center"),
        rx.spacer(),
        rx.text("Vision", font_weight="bold"),
        align="center",
        padding="0.75rem 1rem",
        border_bottom="1px solid var(--gray-5)",
        bg="var(--gray-2)",
        gap="0.75rem",
    )


def option_card(option) -> rx.Component:
    return rx.box(
        rx.text(option.name, font_weight="medium"),
        rx.cond(
            option.description is not None,
            rx.text(option.description, color="gray"),
            rx.box(),
        ),
        option_control(option),
        border="1px solid var(--gray-5)",
        border_radius="8px",
        padding="0.75rem",
        width="100%",
    )


def option_control(option) -> rx.Component:
    return (
        rx.match(
            option.type,
            (
                "checkbox",
                rx.flex(
                    rx.el.input(
                        type="checkbox",
                        default_checked=option.value,
                    ),
                    align="center",
                    gap="0.5rem",
                ),
            ),
            (
                "value_selector",
                rx.el.select(
                    rx.foreach(option.values, lambda value: rx.el.option(value, value=value)),
                    default_value=option.value,
                    style={"width": "100%", "padding": "0.25rem"},
                ),
            ),
            (
                "number_field",
                rx.el.input(
                    type="number",
                    default_value=option.value,
                    min=option.min_value,
                    max=option.max_value,
                    style={"width": "100%", "padding": "0.25rem"},
                ),
            ),
            (
                "field",
                rx.el.input(type="text", default_value=option.value, style={"width": "100%", "padding": "0.25rem"}),
            ),
        ),
    )


def method_details() -> rx.Component:
    return rx.vstack(
        rx.heading(
            rx.cond(
                AppState.selected_method,
                AppState.selected_method.name,
                "Select a method",
            ),
            size="4",
        ),
        rx.text(AppState.selected_method.description, color="gray"),
        spacing="4",
        width="100%",
        align="start",
    )


def method_options_sidebar() -> rx.Component:
    return rx.vstack(
        rx.heading("Method options", size="3"),
        rx.cond(
            AppState.selected_method_options.length() > 0,
            rx.foreach(
                AppState.selected_method_options,
                option_card,
            ),
            rx.text("No options available.", color="gray"),
        ),
        spacing="4",
        width="100%",
        align="stretch",
    )


def main_content() -> rx.Component:
    return rx.flex(
        rx.box(
            method_details(),
            flex="1",
            padding="1.5rem",
            width="100%",
            min_h="0",
            height="100%",
            min_w="0",
            overflow_y="auto",
        ),
        rx.box(
            method_options_sidebar(),
            width="160px",
            min_w="160px",
            max_w="160px",
            padding="1.5rem",
            border_left="1px solid var(--gray-5)",
            bg="var(--gray-1)",
            min_h="0",
            height="100%",
            overflow_y="auto",
            flex_shrink="0",
        ),
        flex="1",
        width="100%",
        min_h="0",
        height="100%",
    )


def footer() -> rx.Component:
    return rx.flex(
        rx.text(AppState.last_menu_action, font_size="1", color="gray"),
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
                AppState.navbar_collapsed,
                rx.box(width="0px"),
                navbar(),
            ),
            rx.box(
                main_content(),
                footer(),
                display="flex",
                flex_direction="column",
                flex="1",
                min_h="0",
            ),
            width="100%",
            flex="1",
            min_h="0",
        ),
        display="flex",
        flex_direction="column",
        align="start",
        height="100vh",
        bg="var(--gray-2)",
    )


def index() -> rx.Component:
    return app_shell()


app = rx.App()
app.add_page(index, route="/", title="Image Transform Panel")
# app.compile()
