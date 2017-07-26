from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.ising_2d import Ising_2d
import tensorflow as tf

def main(_):
  I = Ising_2d(lattice_size=32, num_ensemble=100, num_iter=50000,
               J=1.0, B=0.0, T=1.0)
  I.build_graph()
  I.run_session()
  
if __name__ == '__main__':
  tf.app.run()
