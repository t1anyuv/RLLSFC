import logging
from pathlib import Path
from typing import Optional


def setup_logging(
    log_name: str, 
    verbose: bool = False,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """设置日志记录器
    
    Args:
        log_name: 日志名称
        verbose: 是否详细输出
        log_file: 日志文件路径（可选）
    """
    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger(log_name)
    logger.setLevel(level)

    # 清除已有的handlers
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
