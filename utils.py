import logging
import os

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename=os.path.join('logs', 'training.log'),
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
    )

def log(message):
    print(message)
    logging.info(message)
