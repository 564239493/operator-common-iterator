# -*- coding: UTF-8 -*-
"""
功能：日志模块
版权信息：华为技术有限公司，版本所有(C) 2022-2022
修改记录：2023/12/6 16:23
"""

import logging
import threading
from logging import handlers


class SingletonType(type):
    _instance_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            with SingletonType._instance_lock:
                if not hasattr(cls, "_instance"):
                    cls._instance = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instance


class Logger(metaclass=SingletonType):
    def __init__(self, **kwargs):
        self.level = kwargs.get("log_level", logging.INFO)
        self.fmt = kwargs.get("format",
                              "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s")
        self.console = kwargs.get("console", True)
        self.file = kwargs.get("file", None)
        self.when = kwargs.get("when", "D")
        self.back_count = kwargs.get("back_count", 365)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.level)
        self.format = logging.Formatter(self.fmt)

        if self.console:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(self.format)
            self.logger.addHandler(stream_handler)

        if self.file:
            time_handler = handlers.TimedRotatingFileHandler(filename=self.file, when=self.when,
                                                             backupCount=self.back_count, encoding="utf-8")
            time_handler.setFormatter(self.format)
            self.logger.addHandler(time_handler)
