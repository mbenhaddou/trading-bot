import logging, sys
from typing import Literal

def setup_logging(level: Literal["DEBUG","INFO","WARNING","ERROR","CRITICAL"]="INFO"):
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log")
    ]
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers
    )
