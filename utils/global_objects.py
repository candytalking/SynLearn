from dataloader.data_generator import DataGenerator
from config.exp_config import ExpConfig
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
data_generator = DataGenerator()
exp_config = ExpConfig()
