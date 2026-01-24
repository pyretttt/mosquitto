import logging
from logging.handlers import RotatingFileHandler

last_used_logger = logging.Logger(name="last_used_screen")
last_used_logger.setLevel(logging.INFO)
last_used_formatter = logging.Formatter("%(asctime)s: %(message)s")
file_handler = RotatingFileHandler("nicegui_app/no_index/last_used_screen", mode="a", maxBytes=8192, delay=False)
file_handler.setFormatter(last_used_formatter)
file_handler.setLevel(logging.INFO)
last_used_logger.addHandler(file_handler)
