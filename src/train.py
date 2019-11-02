from glob import glob
from model import Colorizer
import tensorflow as tf

print('TensorFlow', tf.__version__)
print('Executing eagerly =>', tf.executing_eagerly())
tf.config.optimizer.set_jit(True)

config = {
    'distribute_strategy': tf.distribute.OneDeviceStrategy(device='/gpu:0'),
    'epochs': 100,
    'batch_size': 32,
    'd_lr': 3e-5,
    'g_lr': 3e-4,
    'image_list': glob('wiki/*/*'),
    'model_dir': 'model_files',
    'tensorboard_log_dir': 'logs',
    'checkpoint_prefix': 'ckpt',
    'restore_parameters': False,
    'mode': 'train'
}
colorizer = Colorizer(config)
colorizer.train()
