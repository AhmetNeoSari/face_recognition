from loguru import logger
from dataclasses import dataclass

@dataclass
class Logger:
    log_file: str
    level: str
    rotation: str
    retention: str
    compression: str
    format: str

    def __post_init__(self):
        logger.remove()  # Varsayılan logger'ı kaldırır
        logger.add(
            self.log_file,
            level=self.level,
            rotation=self.rotation, 
            retention=self.retention,
            compression=self.compression,
            format=self.format,
        )

    def error(self, message: str):
        logger.opt(depth=1).error(message)

    def warning(self, message: str):
        logger.opt(depth=1).warning(message)
    
    def info(self, message: str):
        logger.opt(depth=1).info(message)
    
    def debug(self, message: str):
        logger.opt(depth=1).debug(message)
    
    def critical(self, message: str):
        logger.opt(depth=1).critical(message)
    
    def trace(self, message: str):
        logger.opt(depth=1).trace(message)