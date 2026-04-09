from openbb import obb

def as_markdown_table(oboject) -> str:
    return (
        "```\n"
        + oboject.to_df().T
            .rename_axis("Field")
            .rename(columns={0: "Value"})
            .to_markdown()
        + "\n```"
    )