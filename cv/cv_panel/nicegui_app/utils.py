import base64


def bytes_to_data_url(data: bytes, filename: str | None = None) -> str:
    """Convert bytes into a data URL; best effort for content type from filename."""
    content_type = "image/png"
    if filename and "." in filename:
        ext = filename.lower().rsplit(".", 1)[-1]
        if ext in ("jpg", "jpeg"):
            content_type = "image/jpeg"
        elif ext in ("png",):
            content_type = "image/png"
        elif ext in ("gif",):
            content_type = "image/gif"
        elif ext in ("webp",):
            content_type = "image/webp"
        elif ext in ("bmp",):
            content_type = "image/bmp"
        elif ext in ("tif", "tiff"):
            content_type = "image/tiff"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{content_type};base64,{b64}"
