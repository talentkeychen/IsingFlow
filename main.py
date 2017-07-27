from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.ising_2d import Ising_2d
import tensorflow as tf
import numpy as np

def main(_):
  T = np.linspace(1, 4, 20)
  E = []
  M = []
  C = []
  X = []
  for t in T:
    I = Ising_2d(lattice_size=32, num_ensemble=10, num_iter=50000,
                 J=1.0, B=0.0, T=t)
    I.build_graph()
    _E, _M, _C, _X = I.run_session()
    E.append(_E)
    M.append(_M)
    C.append(_C)
    X.append(_X)
  
if __name__ == '__main__':
  tf.app.run()
