# -*- coding: utf-8 -*-
"""
Data Class Module
"""

import numpy as np           # Work with matrices (arrays)
import pandas as pd          # Load csv files
import math                  # Math Operations

class DATA():
	'Class that hold input and outputs of a DataSet'
	
	def __init__(self):
	
		# Model Hyper parameters
		self.problem = None           # which problem will be solved
		self.problem_detail = None    # More details about a specific data set
		self.norm_type = None         # Type of normalization
		self.label_type = None        # type of label codification
		self.hold_method = None       # type of hold out
		self.train_size = None        # Percentage of samples for training
		
		# Data in "NumPy and Pandas Away"
		self.df = None       # Dataframe
		self.input = None    # Input Matrix
		self.output = None   # Output Matrix
		self.lbl = None      # Original labels
		self.N = None        # Number of Samples
		self.p = None        # Number of Attributes
		self.Nc = None       # Number of Classes
		
		# Data Statistics
		self.Xmin = None 	# Minimum Value of inputs
		self.Xmax = None 	# Maximum Value of inputs
		self.Xmean = None 	# Mean Value of inputs
		self.Xstd = None 	# Standard Deviation of inputs
		
		# Hold out Data
		self.X_tr = None     # Training input Matrix
		self.X_ts = None     # Test input Matrix
		self.y_tr = None     # Training Output Matrix
		self.y_ts = None     # Test Output Matrix
	
	def class_loading(self,problem='iris',problem_detail=1):
		
		self.problem = problem
		self.problem_detail = problem_detail
		
		if (self.problem =='iris'):
			# load data from file
			self.df = pd.read_csv("iris.data.csv")
			# load input
			self.input = np.array(self.df.drop('classe',1))
			# load labels
			self.lbl = self.df['classe'].unique()
			# load attributes
			self.N = self.input.shape[0]
			self.p = self.input.shape[1]
			self.Nc = self.df['classe'].nunique()
			# load output
			out = pd.get_dummies(self.df['classe'])     # dataframe
			out = np.array(out.values)                  # numpy array
			out = out.astype('int32')         		    # integer values
			# Adjust output
			out = np.argmax(out,axis=1).T + 1 	        # sequential values
			self.output = out.reshape(self.N,1)         # adjust shape
#			for i in range(self.N):
#				for j in range (self.Nc):
#					if (self.output[i,j] != 1):
#						self.output[i,j] = -1
		else:
			print('type an existing dataset')
	
	def normalize(self,norm_type='zscore'):
		
		# Hold Type of Normalization
		self.norm_type = norm_type
		
		# Verify if hold out was already done and get values
		if(self.X_tr is None):
			self.Xmin = self.input.min(0)
			self.Xmax = self.input.max(0)
			self.Xmean = self.input.mean(0)
			self.Xstd = self.input.std(0)
		else:
			self.Xmin = self.X_tr.min(0)
			self.Xmax = self.X_tr.max(0)
			self.Xmean = self.X_tr.mean(0)
			self.Xstd = self.X_tr.std(0)
		
		# Normalization - Using all data
		if(self.X_tr is None):
			if(norm_type == 'zscore'):
				self.input = (self.input - self.Xmean)/self.Xstd
			elif(norm_type == 'binary'):
				self.input = (self.input - self.Xmin)/(self.Xmax - self.Xmin)
			elif(norm_type == 'bipolar'):
				self.input = 2*(self.input - self.Xmin)/(self.Xmax - self.Xmin) - 1
			else:
				print('type an existing normalization type')
		# Normalization - Using Training data
		else:
			if(norm_type == 'zscore'):
				self.X_tr = (self.X_tr - self.Xmean)/self.Xstd
				self.X_ts = (self.X_ts - self.Xmean)/self.Xstd
			elif(norm_type == 'binary'):
				self.X_tr = (self.X_tr - self.Xmin)/(self.Xmax - self.Xmin)
				self.X_ts = (self.X_ts - self.Xmin)/(self.Xmax - self.Xmin)
			elif(norm_type == 'bipolar'):
				self.X_tr = 2*(self.X_tr - self.Xmin)/(self.Xmax - self.Xmin) - 1
				self.X_ts = 2*(self.X_ts - self.Xmin)/(self.Xmax - self.Xmin) - 1
			else:
				print('type an existing normalization type')
		
	def denormalize(self):
		
		# Denormalization - Using all data
		if(self.X_tr is None):
			if(self.norm_type == 'zscore'):
				self.input = self.input * self.Xstd + self.Xmean
			elif(self.norm_type == 'binary'):
				self.input = self.input * (self.Xmax - self.Xmin) + self.Xmin
			elif(self.norm_type == 'bipolar'):
				self.input = 0.5*(self.input + 1) * (self.Xmax - self.Xmin) + self.Xmin
			else:
				print('type an existing normalization type')
		else:
			if(self.norm_type == 'zscore'):
				self.X_tr = self.X_tr * self.Xstd + self.Xmean
				self.X_ts = self.X_ts * self.Xstd + self.Xmean
			elif(self.norm_type == 'binary'):
				self.X_tr = self.X_tr * (self.Xmax - self.Xmin) + self.Xmin
				self.X_ts = self.X_ts * (self.Xmax - self.Xmin) + self.Xmin
			elif(self.norm_type == 'bipolar'):
				self.X_tr = 0.5*(self.X_tr + 1) * (self.Xmax - self.Xmin) + self.Xmin
				self.X_ts = 0.5*(self.X_ts + 1) * (self.Xmax - self.Xmin) + self.Xmin
			else:
				print('type an existing normalization type')

	def label_encode(self,label_type='bipolar'):
		
		# Hold Type of Label Codification
		self.label_type = label_type
		
		if(label_type == 0):
			self.output = self.output
		elif(label_type == 'sequential'):
			# ToDo - All
			self.output = self.output
		elif(label_type == 'binary'):
			# ToDo - All
			self.output = self.output
		elif(label_type == 'bipolar'):
			out = -1*np.ones((self.N,self.Nc))
			out = out.astype('int32')
			for i in range(self.N):
				out[i,self.output[i]-1] = 1
			self.output = out
		else:
			print('type an existing label codification type')
	
	def hold_out(self,hold_method='aleatory',train_size='0.80'):
		
		# Hold out method and percentage of data for training
		self.hold_method = hold_method
		self.train_size = train_size
		
		# Get Input, Output and number of samples
		X = self.input
		Y = self.output
		N = self.N
		
		if(hold_method == 'aleatory'):
			
			# shuffle data
			I = np.random.permutation(N)
			X = X[I,:]
			Y = Y[I,:]
			
			# Number of samples for training
			Ntr = math.floor(N*train_size)
			
			# Hold samples for training and test
			self.X_tr = X[0:Ntr,:]
			self.X_ts = X[Ntr:,:]
			self.y_tr = Y[0:Ntr,:]
			self.y_ts = Y[Ntr:,:]
#		elif(hold_method == 'min_class'):
			# ToDo - All
		else:
			print('type an existing holdout_method')

# End of File