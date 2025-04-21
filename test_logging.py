# test_logging.py
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
    force=True
)

for i in range(10):
    logging.info(f"Step {i}")
    time.sleep(1)
