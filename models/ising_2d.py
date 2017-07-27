from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.model_base import Ising
from tqdm import tqdm
from six.moves import xrange

import tensorflow as tf
import numpy as np

DTYPE=tf.float32

def _periodic_padding(lattice, padding=1):
  '''
  Create a periodic padding (wrap) around the lattice, to emulate periodic boundary conditions
  From https://github.com/tensorflow/tensorflow/issues/956, FedericoMuciaccia suggestion
  '''
  
  upper_pad = lattice[:,-padding:, :, :]
  lower_pad = lattice[:, :padding, :, :]
  
  partial_lattice = tf.concat([upper_pad, lattice, lower_pad], axis=1)
  
  left_pad = partial_lattice[:, :, -padding:, :]
  right_pad = partial_lattice[:, :, :padding, :]
  
  padded_lattice = tf.concat([left_pad, partial_lattice, right_pad], axis=2)
  
  return padded_lattice


class Ising_2d(Ising):
  
  # let Hamiltonian as \beta H = - K \sum_{<i,j>} \sigma_i \sigma_j - L \sum_{i} sigma_i
  def __init__(self, lattice_size, num_ensemble, num_iter, time_avg=1000, J=1.0, B=0.0, T=1.0, periodic=True, seed=531):
    assert lattice_size % 2 == 0
    assert time_avg > 0
    self.lattice_size = lattice_size
    self.num_ensemble = num_ensemble
    self.num_iter = num_iter
    self.J = J # interaction
    self.B = B # Magnetic field
    self.T = T # Temperature scaled with boltzmann factor T <- k_{B} T, therefore \beta = 1 / T
    self.periodic = True
    self.seed = seed
    self.init = tf.constant_initializer(2 * np.random.randint(0, 2, [num_ensemble, lattice_size, lattice_size, 1]) - 1,
                                        dtype=DTYPE)
    self.graph = tf.Graph()
    self.build_graph(time_avg)
    
  def _Hamiltonian(self, lattice, J, B) :
    # let Hamiltonian as H = - J \sum_{<i,j>} \sigma_i \sigma_j - B \sum_{i} sigma_i
    _filter = tf.constant([[[[0]], [[1]], [[0]]],
                          [[[1]], [[0]], [[1]]],
                          [[[0]], [[1]], [[0]]]], dtype=DTYPE, name='kernel')
    if self.periodic:
      padded = _periodic_padding(lattice, padding=1)
      conv = tf.nn.conv2d(padded, filter=_filter, strides=[1, 1, 1, 1], padding='VALID')
      conv = tf.multiply(conv, lattice) / 2
    else:
      conv = tf.nn.conv2d(lattice, filter=_filter, strides=[1, 1, 1, 1], padding='SAME')
      conv = tf.multiply(conv, lattice) / 2 # TODO : This is not exact.
      
    Hamiltonian = - J * tf.reduce_sum(conv, axis=[1,2,3]) - B * tf.reduce_sum(lattice, axis=[1,2,3])
    
    return Hamiltonian
    
  def _Metropolis(self, lattice, J, B, T):
    _filter = tf.constant([[[[0]], [[1]], [[0]]],
                           [[[1]], [[0]], [[1]]],
                           [[[0]], [[1]], [[0]]]], dtype=DTYPE, name='kernel')
    
    a = np.zeros(shape=[self.lattice_size, self.lattice_size, 1], dtype=np.bool)
    for i in xrange(self.lattice_size):
      for j in xrange(self.lattice_size):
        a[i,j,0] = True if (i+j) % 2 == 0 else False
        
    _even_filter = tf.constant(a, dtype=tf.bool)
    
    def delta_energy(lattice):
      if self.periodic:
        padded = _periodic_padding(lattice, padding=1)
        conv = tf.nn.conv2d(padded, filter=_filter, strides=[1, 1, 1, 1], padding='VALID')
        conv = tf.multiply(conv, lattice)
      else:
        conv = tf.nn.conv2d(lattice, filter=_filter, strides=[1, 1, 1, 1], padding='SAME')
        conv = tf.multiply(conv, lattice) # TODO : This is not exact.

      conv = J * conv + B * lattice
      return 2*conv
    
    delta_even = delta_energy(lattice)
    random = tf.random_uniform(delta_even.get_shape().as_list(), maxval=1, seed=self.seed)
    condition_1 = tf.less(delta_even, 0)
    condition_2 = tf.less(random,
                          tf.exp(-delta_even / T))
    cond = tf.logical_or(condition_1, condition_2)
    condition = tf.logical_and(_even_filter, cond)
    
    lattice = tf.where(condition, -lattice, lattice)
    delta_odd = delta_energy(lattice)
    
    condition_1 = tf.less(delta_odd, 0)
    condition_2 = tf.less(random,
                          tf.exp(-delta_odd / T))
    cond = tf.logical_or(condition_1, condition_2)
    condition = tf.logical_and(tf.logical_not(_even_filter), cond)
    
    return tf.get_variable('lattice').assign(tf.where(condition, -lattice, lattice))
  
    
  def build_graph(self, time_avg):
    with self.graph.as_default() as g:
      _lattice_size = self.lattice_size
      _num_ensemble = self.num_ensemble
      _seed = self.seed
      
      T = tf.Variable(self.T, name='Temperature', dtype=DTYPE)
      J = tf.Variable(self.J, name='Interaction', dtype=DTYPE)
      B = tf.Variable(self.B, name='Field', dtype=DTYPE)
      
      lattice = tf.get_variable('lattice',
                                shape=[_num_ensemble, _lattice_size, _lattice_size, 1],
                                dtype=DTYPE,
                                initializer=self.init)
      
      moving_avg = tf.train.ExponentialMovingAverage(1.0 - 1/time_avg)

      M = tf.reduce_mean(lattice, axis=[1,2,3])
      M_square = tf.reduce_mean(tf.square(tf.reduce_sum(lattice, axis=[1,2,3])))
      
      H = self._Hamiltonian(lattice, J, B)
      H_square = tf.reduce_mean(tf.square(H))
      H = tf.reduce_mean(H)
      
      avg_op = moving_avg.apply([H, H_square, M, M_square])
      self.energy = moving_avg.average(H)
      self.energy_square = moving_avg.average(H_square)
      self.magnetization = tf.reduce_mean(tf.abs(moving_avg.average(M)))
      self.magnetization_square = moving_avg.average(M_square)

      self.specific_heat = tf.div(self.energy_square - tf.square(self.energy),
                                  tf.square(T) * self.lattice_size**2,
                                  name='specific_heat')

      self.susceptibility = tf.div(self.magnetization_square - tf.square(self.magnetization * self.lattice_size**2),
                                   tf.multiply(T, self.lattice_size**2),
                                   name='susceptibility')
      
      tf.get_variable_scope().reuse_variables()
      with tf.control_dependencies([avg_op]):
        self.step = self._Metropolis(tf.get_variable('lattice'), J, B, T)
  
  def run_session(self, temperatures=None):
    with tf.Session(graph=self.graph) as sess:
      tf.global_variables_initializer().run()
      
      E, M, C, X = [], [], [], []
      
      if temperatures is None:
        temperatures = [self.T]
      for i in tqdm(temperatures, desc='Temperature'):
        feed = {'Temperature:0' : i}
        for _ in tqdm(xrange(self.num_iter), desc='Metropolis'):
          _, _E, _E2, _M, _M2, _C, _X = sess.run([self.step, self.energy, self.energy_square,
                                                  self.magnetization, self.magnetization_square,
                                                  self.specific_heat, self.susceptibility],
                                                 feed_dict=feed)
        E.append(_E / self.lattice_size**2)
        M.append(_M)
        C.append(_C)
        X.append(_X)
        
      return E, M, C, X
