import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("test.log"),  # fichier de log
        logging.StreamHandler()           # console
    ]
)

# Test des messages
logging.debug("Message DEBUG – ne s'affiche pas avec level=INFO")
logging.info("Message INFO – normal")
logging.warning("Message WARNING – attention")
logging.error("Message ERROR – problème")
logging.critical("Message CRITICAL – critique")
