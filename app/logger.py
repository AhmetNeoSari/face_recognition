from loguru import logger
from dataclasses import dataclass

@dataclass
class Logger:
    name: str
    log_file: str #TODO logfile, level,rotation bakılacak error,info vs fonk ekle hepsi aynı loggerı kullancak nesneyi gönder
    level: str = "INFO"
    rotation: str = "5 MB"
    retention: str = "10 days"
    compression: str = "zip"
    
    def __post_init__(self):
        logger.remove()
        logger.add(
            self.log_file,
            level=self.level,
            rotation=self.rotation,
            retention=self.retention,
            compression=self.compression,
            format="{time} {level} {message}", #FORMAT Değişecek Time, line vb. eklenecek
            enqueue=True,
            backtrace=True,
            diagnose=True
        )
        self.logger = logger.bind(name=self.name)

    #TODO depth
    def get_logger(self):
        return self.logger
    
    def error(self, message: str):
        self.logger.error(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def critical(self, message: str):
        self.logger.critical(message)
    
    def trace(self, message: str):
        self.logger.trace(message)
