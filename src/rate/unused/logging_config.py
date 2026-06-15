import logging
from datetime import datetime
from pathlib import Path


def setup_logging(name: str, save_dir: str = "logs", filename: str = None) -> logging.Logger:
    """Setup logging configuration for a given module.

    Args:
        name: Name of the module/logger
        save_dir: Directory to save log files
        filename: Optional filename to include in log name

    Returns:
        Configured logger instance
    """
    # Create logs directory in save_dir if it doesn't exist
    log_dir = Path(save_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Build log filename with optional input filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if filename:
        # Extract base filename without extension
        base_filename = Path(filename).stem
        log_filename = f"{name}_{base_filename}_{timestamp}.log"
    else:
        log_filename = f"{name}_{timestamp}.log"

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_dir / log_filename)

    # Create formatters and add it to handlers
    log_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
