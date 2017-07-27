from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.ising_2d import Ising_2d
from tqdm import tqdm

import _pickle as pickle
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('data_dir', '/tmp/ising_flow',
                       'The directory where results are stored.')

tf.flags.DEFINE_integer('start_lattice_size', 16,
                        'Starting lattice size of Ising model.')

tf.flags.DEFINE_integer('num_system', 5,
                        'Total number of lattice system')

tf.flags.DEFINE_integer('lattice_factor', 2,
                        'Lattice size multiply factor for increase size')

tf.flags.DEFINE_integer('num_ensemble', 1,
                        'Number of ensemble.')

tf.flags.DEFINE_integer('num_iter', 10000,
                        'Number of iteration for Metropolis algorithm.')

tf.flags.DEFINE_integer('time_avg', 1000,
                        'Time average interval for average over time.')

tf.flags.DEFINE_integer('num_temperature', 20,
                        'Number of temperature points.')

tf.flags.DEFINE_float('t_start', 1.0,
                      'Starting temperature.')

tf.flags.DEFINE_float('t_end', 4.0,
                      'End temperature.')


def main(_):
  lattices = [FLAGS.start_lattice_size * FLAGS.lattice_factor ** (i+1) for i in range(FLAGS.num_system)]
  temperatures = np.linspace(FLAGS.t_start, FLAGS.t_end, FLAGS.num_temperature)
  
  for l in tqdm(lattices, desc='Lattice'):
    I = Ising_2d(lattice_size=l,
                 num_ensemble=FLAGS.num_ensemble,
                 num_iter=FLAGS.num_iter,
                 time_avg=FLAGS.time_avg)
    E, M, C, X = I.run_session(temperatures)
    
    data = {}
    data['Temperature'] = temperatures
    data['Energy'] = np.array(E)          # energy per spin
    data['Magnetization'] = np.array(M)   # Magnetization
    data['Heat_capacity'] = np.array(C)   # Heat capacity per spin
    data['Susceptibility'] = np.array(X)  # Susceptibility per spin
    
    if not tf.gfile.Exists(FLAGS.data_dir):
      tf.gfile.MakeDirs(FLAGS.data_dir)
    
    with open(FLAGS.data_dir + '/lattice_%d.pkl' % l, 'wb') as f:
      pickle.dump(data, f)
  
if __name__ == '__main__':
  tf.app.run()
