from src.config import Config
from src.utils import create_folder
from src import models, dataset, trainer, betacal

# get all configurations
config = Config().get_yaml()

# create logs and results folder, if they do not exist
create_folder(f"{config['paths'].get('logs')}")
create_folder(f"{config['paths'].get('results')}")
