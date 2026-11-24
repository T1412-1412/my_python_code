import logging
import os
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler


def setup_logger(name: str = "agent", log_level: str = "INFO") -> logging.Logger:
    """
    创建并配置统一格式的日志器
    :param name: 日志器名称，区分不同模块
    :param log_level: 日志级别，可选 DEBUG/INFO/WARNING/ERROR
    :return: 配置好的 logger 对象
    """
    logger = logging.getLogger(name)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 日志级别校验
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    level = log_level.upper()
    if level not in valid_levels:
        level = "INFO"

    # 总开关设为最低的DEBUG，让各handler自己控制级别
    logger.setLevel(logging.DEBUG)

    # ========== 日志格式 ==========
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ========== 控制台输出处理器 ==========
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.encoding = "utf-8"
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)  # 控制台按传入级别输出
    logger.addHandler(console_handler)

    # ========== 文件输出处理器（按天切割） ==========
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "logs"
    )
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"agent_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=30,  # 保留30天
        encoding="utf-8"
    )
    file_handler.suffix = "%Y%m%d"
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # 文件记录更详细的DEBUG日志
    logger.addHandler(file_handler)

    return logger


# 全局默认日志器
logger = setup_logger("agent", "INFO")
