import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib import lines

def compute_rms(truth,error,size):
   return np.sqrt(sum(np.square(truth-error)/size))

class random_walk_sequence:
   IDX_ARB_LABEL = 0
   def __init__(self, dataset_file):
      self.dataset_file = dataset_file
      self.data = np.array([])
      self.label = float('inf')
      self.raw_data = np.loadtxt(self.dataset_file)
      self.data = self.raw_data[:][:-1]
      self.label = self.raw_data[-1][0]
      self.data_point_count,_ = self.data.shape

class run_training_on_data_set:
   def __init__(self,training_set_path):
      self.path = training_set_path
      self.sequences = []
      self.weights = []
      # Find all of the sequence files:
      for dirpath, dnames, fnames in os.walk(self.path):
         for f in fnames:
            self.sequences.append(random_walk_sequence(os.path.join(self.path,f)))
   def train(self,alpha,lambda_,sigma,experiment_number,initial_weights):
      _,num_states = self.sequences[0].data.shape
      weights = np.copy(initial_weights)
      converged = False
      if experiment_number == 2:
         # Second experiment is single pass through of data
         max_iter = 1
      else:
         max_iter = 20
      counter = 0
      deltaWt = np.array([0.0]*num_states)
      while (not converged) and (counter<max_iter):
         counter+=1
         weights_old = weights.copy()
         for seq in self.sequences:
            deltaWt = np.array([0.0]*num_states)
            for t in range(seq.data_point_count):
               if experiment_number == 2:
                  deltaWt = np.array([0.0]*num_states)
               Pt = np.dot(seq.data[t][:],weights)
               if t+1 == seq.data_point_count:
                  Ptp1 = seq.label
               else:
                  Ptp1 = np.dot(seq.data[t+1][:],weights)
               summation = np.array([0.0]*num_states)
               for k in range(t+1):
                  # set discounted lambda:
                  lambda_discounted = lambda_**(t-k)
                  summation+=lambda_discounted*seq.data[k][:]
               deltaWt += alpha*(Ptp1-Pt)*summation
               if experiment_number == 2:
                  weights+=deltaWt
            if experiment_number == 1:
               weights+=deltaWt
         if compute_rms(weights_old,weights,5) < sigma:
            converged = True
      self.weights = weights

if __name__ == '__main__':
   # Setup Variables and get the datasets:
   dir_path_training_data = 'training_data'
   training_set_dirs = []
   for dirpath, dnames, fnames in os.walk(dir_path_training_data):
      if len(dnames) > 0:
         # This is the list of dirs for the first level down and all I care about:
         training_set_dirs = dnames

   # Study Variables:
   figure3 = True
   figure4 = True
   figure5 = True
   rms_unit_test = False
   debug = False # Flag to add debug output
   inspect_plots = True

   # Common Variables:
   expected_weights = np.array([1/6,2/6,3/6,4/6,5/6])

   # Experiment 1, Figure 1:
   if figure3:
      print("Running Experiment 1, Generating Figure 3")
      experiment_number = 1
      sigma = 0.01
      num_train = 100
      lambdas = [0.0,0.1,0.3,0.5,0.7,0.9,1.0]
      alpha_schedule = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
      avr_rms_error = []
      for lambda_value,alpha_value in zip(lambdas,alpha_schedule):
         count = 0
         agregate_weights = np.array([0.0]*5)
         for training_set in training_set_dirs:
            count+=1
            path = os.path.join(dir_path_training_data,training_set)
            data_set = run_training_on_data_set(path)
            data_set.train(alpha_value,lambda_value,sigma,experiment_number,np.array([0.5]*5))
            agregate_weights += data_set.weights
            if count >= num_train:
               break
         agregate_weights_trimmed = agregate_weights/num_train
         if debug:
            print(agregate_weights_trimmed)
         rms_error = compute_rms(agregate_weights_trimmed,expected_weights,5)
         avr_rms_error.append(rms_error)
      fig = plt.figure()
      ax = fig.add_subplot(111)
      plt.plot(lambdas,avr_rms_error,'ko-')
      # Annotate the Widrow-Hoff Point
      ax.annotate('Widrow-Hoff', xy=(lambdas[-1]-0.2,avr_rms_error[-1]))
      plt.xlabel('\u03BB',fontsize=14)
      plt.ylabel('ERROR',fontsize=14)
      if inspect_plots:
         plt.show()
      fig.savefig('figure3.png', dpi=1000)
      print("Figure 3 Generated")

   # Experiment 2, Figure 2:
   # Modulate over a range of alpha values:
   if figure4:
      print("Running Experiment 2, Generating Figure 4")
      # Compute initial error (same for all with lambda = 0.0
      init_error = compute_rms(expected_weights, np.array([0.5]*5),5)
      alphas = [0.0,0.1,0.2,0.3,0.4,0.5]
      #alphas = [0.0,0.1,0.2]
      sigma = 0.1
      expected_weights = np.array([1/6,2/6,3/6,4/6,5/6])
      num_train = 1
      experiment_number = 2
      lambdas = [0.0,0.3,0.8,1.0]
      #lambdas = [0.0,0.1,0.3,0.5,0.7,0.9,0.95,1.0]
      fig = plt.figure()
      ax = fig.add_subplot(111)
      for lambda_value in lambdas:
         avr_rms_error = [init_error]
         for alpha in alphas:
            if alpha == 0.0:
               continue
            count = 0
            agregate_weights = np.array([0.0]*5)
            for training_set in training_set_dirs:
               count+=1
               path = os.path.join(dir_path_training_data,training_set)
               data_set = run_training_on_data_set(path)
               data_set.train(alpha,lambda_value,sigma,experiment_number,np.array([0.5]*5))
               agregate_weights += data_set.weights
               if count >= num_train:
                  break
            agregate_weights_trimmed = agregate_weights/num_train
            rms_error = compute_rms(agregate_weights_trimmed,expected_weights,5)
            avr_rms_error.append(rms_error)
         if lambda_value == 1.0:
            legend_label = "\u03BB = "+str(lambda_value)+"\n(Widrow-Hoff)"
         else:
            legend_label = "\u03BB = "+str(lambda_value)
         plt.plot(alphas,avr_rms_error,'k-o',label=legend_label)
         if lambda_value == 0.3:
            ax.annotate(legend_label, xy=(alphas[-1]-0.04,avr_rms_error[-1]+0.02))
         elif lambda_value == 1.0:
            ax.annotate(legend_label, xy=(alphas[-1]-0.09,avr_rms_error[-1]-0.06))
         else:
            ax.annotate(legend_label, xy=(alphas[-1]-0.04,avr_rms_error[-1]-0.035))
      #plt.legend(loc="upper left")
      plt.xlabel('\u03B1',fontsize=14)
      plt.ylabel('ERROR',fontsize=14)
      if inspect_plots:
         plt.show()
      fig.savefig('figure4.png', dpi=1000)
      print("Figure 4 Generated")

   if figure5:
      print("Running Experiment 2, Generating Figure 5")
      experiment_number = 2
      sigma = 0.01
      num_train = 100
      lambdas = [0.0,0.1,0.3,0.5,0.7,0.9,0.95,1.0]
      alpha_schedule = [0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1]
      avr_rms_error = []
      for lambda_value,alpha_value in zip(lambdas,alpha_schedule):
         count = 0
         agregate_weights = np.array([0.0]*5)
         for training_set in training_set_dirs:
            count+=1
            path = os.path.join(dir_path_training_data,training_set)
            data_set = run_training_on_data_set(path)
            data_set.train(alpha_value,lambda_value,sigma,experiment_number,np.array([0.5]*5))
            agregate_weights += data_set.weights
            if count >= num_train:
               break
         agregate_weights_trimmed = agregate_weights/num_train
         if debug:
            print(agregate_weights_trimmed)
         rms_error = compute_rms(agregate_weights_trimmed,expected_weights,5)
         avr_rms_error.append(rms_error)
      fig = plt.figure()
      ax = fig.add_subplot(111)
      plt.plot(lambdas,avr_rms_error,'ko-')
      plt.xlabel('\u03BB',fontsize=14)
      plt.ylabel('ERROR (Best \u03B1)',fontsize=14)
      if inspect_plots:
         plt.show()
      fig.savefig('figure5.png', dpi=1000)
      print("Figure 5 Generated")

   # Sandbox
   if rms_unit_test:
      truth = np.array([1/6,2/6,3/6,4/6,5/6])
      initial = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
      rms = compute_rms(truth,initial,len(initial))
      print(rms)
      # Should be around 0.23-0.25