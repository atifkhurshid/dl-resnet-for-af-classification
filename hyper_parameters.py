# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
import tensorflow as tf
import datetime

FLAGS = tf.app.flags.FLAGS

# The following flags are related to save paths, tensorboard outputs and screen outputs

tf.app.flags.DEFINE_string('version', 'resnet_1009', '''A version number defining the directory to save logs and checkpoints''')
tf.app.flags.DEFINE_integer('report_freq',1 , '''Steps taken to write summaries''')
tf.app.flags.DEFINE_integer('print_freq',50, '''Steps taken to output accuracy on screen ''')
tf.app.flags.DEFINE_float('train_ema_decay', 0.95, '''The decay factor of the train error's moving average shown on tensorboard''')
tf.app.flags.DEFINE_float('dropout', 0.5, '''Dropout propotion''')


# The following flags define hyper-parameters regards training

tf.app.flags.DEFINE_integer('train_steps',2000 , '''Total steps that you want to train''')
tf.app.flags.DEFINE_integer('train_batch_size', 64, '''Train batch size''')
# DO NOT CHANGE THIS
tf.app.flags.DEFINE_boolean('is_full_validation', False, '''Validation w/ full validation set or a random batch''')
tf.app.flags.DEFINE_boolean('use_dropout', False, '''Use dropout or not''')
# >>>>>>>> Specify a large enough validation set size for full validation
tf.app.flags.DEFINE_integer('validation_reserved_slidesets_number', 16, '''How many slidesets (directories) should be reserved for validation. Note that there are only 82 slidesets in total''')
tf.app.flags.DEFINE_integer('validation_total_batch_size',1000 , '''Total Validation batch size for intermediate report''')
tf.app.flags.DEFINE_integer('validation_batch_size', 1000, '''Validation batch size, must be a divisor of validation_total_batch_size''')
tf.app.flags.DEFINE_integer('test_batch_size',1000, '''Test batch size''')

tf.app.flags.DEFINE_float('init_lr', 0.05, '''Initial learning rate''')
tf.app.flags.DEFINE_float('lr_decay_factor', 0.5, '''How much to decay the learning rate each time''')
tf.app.flags.DEFINE_integer('decay_step0', 1000, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_step1', 1750, '''At which step to decay the learning rate''')
tf.app.flags.DEFINE_integer('decay_step2', 2500, ''' /////// ''')


# The following flags define hyper-parameters modifying the training network

tf.app.flags.DEFINE_integer('num_residual_blocks', 5 , '''How many residual blocks do you want''')
tf.app.flags.DEFINE_float('weight_decay', 0.0002, '''scale for l2 regularization''')


# The following flags are related to data-augmentation

tf.app.flags.DEFINE_integer('padding_size', 2, '''In data augmentation, layers of zero padding on each side of the image''')
tf.app.flags.DEFINE_boolean('force_grayscale', True, '''Treat color images as grayscale ones. Recommended for machines with limited memory.''')


# If you want to load a checkpoint and continue training
# Create folder ckpt before running training
tf.app.flags.DEFINE_string('ckpt_path', 'ckpt/model.ckpt-100000', '''Checkpoint directory to restore''')

train_dir = 'logs_' + FLAGS.version + '_' + str(FLAGS.magn) + 'x' + '_' + datetime.datetime.now().strftime("%m%d%H%M") + '/'
