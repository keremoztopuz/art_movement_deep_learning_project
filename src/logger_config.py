import logging
import os
from datetime import datetime

def setup_logger(name="art_classifier", log_dir="logs"):
    """
    Logger konfigürasyonu
    
    Kullanım:
        from logger_config import setup_logger
        logger = setup_logger()
        logger.info("Eğitim başladı")
        logger.warning("Val loss artıyor")
        logger.error("Hata oluştu")
    """
    
    # Log klasörü oluştur
    os.makedirs(log_dir, exist_ok=True)
    
    # Dosya adı (tarih-saat ile)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    # Logger oluştur
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Dosya handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Handler'ları ekle
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger başlatıldı. Log dosyası: {log_file}")
    
    return logger


# Örnek kullanım
if __name__ == "__main__":
    logger = setup_logger()
    
    logger.debug("Debug mesajı (sadece dosyaya)")
    logger.info("Info mesajı")
    logger.warning("Warning mesajı")
    logger.error("Error mesajı")
    logger.critical("Critical mesajı")
    
    # Epoch logging örneği
    for epoch in range(1, 4):
        logger.info(f"Epoch {epoch}/50 | Loss: 1.23 | Val Loss: 1.45 | Val Acc: 0.56")
