from reflex_app.models import Option, NumberField, ValueSelector, Method


method_state = Method(
    name="Image registration",
    description="Using homography, and ransac, to align one image into another",
    options=[
        Option(name="Ransac iterations", info=NumberField(value=20, min_value=1, max_value=1000000)),
        Option(name="Some values", info=ValueSelector(values=["one", "two", "free"])),
    ],
)
