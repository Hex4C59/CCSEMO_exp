import logging
import os


def setup_logger(log_path: str) -> logging.Logger:
    """创建同时输出到控制台和文件的独立 logger（按日志文件名区分）."""
    # 按文件名生成 logger 名称，避免污染 root logger
    logger_name = f"experiment_{os.path.basename(log_path)}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 不向上冒泡到 root，避免重复输出

    # 每次调用都清理旧 handler，确保切换到当前 log_path
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
