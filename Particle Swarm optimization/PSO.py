import random
import numpy as np
import pandas
import matplotlib.pyplot as plt
import math
import time
import progress.bar as Bar
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

start = time.time() #user time
t = time.process_time() # process time

#Amaury Ribeiro

global_best_execution = 0

CODE_VERBOSE = True
DEBUG = False
MINIMIZATION = True #flag to change to maximization (False) or minimization (True)

# to disable ilustration switch these flags
GRAPH_ITERATION_3D = True #build with plotly 3D with iteration saved
GRAPH_SLEEP_3D = False #build with matplot3D and rebuild each iteration

#mathematical expression whose optimal solution is as close to zero as possible
def func_obj(x):  

	n = float(len(x))
	f_exp = -0.2 * math.sqrt(1/n * sum(np.power(x, 2)))

	t = 0
	for i in range(0, len(x)):
		t += np.cos(2 * math.pi * x[i])

	s_exp = 1/n * t
	f = -20 * math.exp(f_exp) - math.exp(s_exp) + 20 + math.exp(1)
    
	return f

#  PSO  
class Particle():

	def __init__(self, n_dimensions, c1, c2, W):

		self.velocity = []
		self.position = []
		self.neighbors = []

		self.bounds = [-2, 2]

		self.c1 = c1 #cognitive constant
		self.c2 = c2 #social constant
		self.W = W
		self.n_dimensions = n_dimensions

		for _ in range(0,n_dimensions):
			self.velocity.append(0)
			self.position.append(random.uniform(self.bounds[0], self.bounds[1]))
		
		self.P_best_position = self.position.copy()

	def evaluate(self):
		return func_obj(self.position)

	def update_velocity(self, G_best_position):

		for i in range(0,self.n_dimensions):

			r1=random.random()
			r2=random.random()
			vel_cognitive = self.c1 * r1 * (self.P_best_position[i]- self.position[i])
			vel_social  = self.c2 * r2 * (G_best_position[i]- self.position[i])
			self.velocity[i] = self.W * self.velocity[i] + vel_cognitive + vel_social

	def update_position(self):
		for i in range(0,self.n_dimensions):
			self.position[i] = self.position[i] + self.velocity[i]

			#max position
			if self.position[i] > self.bounds[1]:
				self.position[i] = self.bounds[1]
			#min position 
			if self.position[i] < self.bounds[0]:
				self.position[i]= self.bounds[0]
		if ( self.evaluate() < func_obj(self.P_best_position)):
			self.P_best_position = self.position.copy()
			

class PSO_Real():

	def __init__(self,n_iterations = 100, n_particles = 100, c1 = 1, c2 = 2, W=0.5, num_dimensions = 3, topology = 'ALL'):

		self.n_particles = n_particles
		global global_best_execution

		if CODE_VERBOSE:  #the more dimensions, the more difficult the problem
			print("Particle Swarm optimization for ", num_dimensions," dimensions...")

		swarm=[]
		for i in range(0,n_particles):
			swarm.append(Particle(n_dimensions = num_dimensions, c1=c1, c2=c2, W=W))
		
		G_best_position = swarm[0].position.copy()
		i=0
		bests = []
		x, y, z = [], [], []

		while i < n_iterations:

			if CODE_VERBOSE:
				print('Iteration ', i)

			if GRAPH_SLEEP_3D: #second iteration 
				if(i < 20):
					fig = plt.figure(figsize=(16,8))
					self.visualize_3D(swarm, fig)
					txt =f"{n_particles} particles - {num_dimensions} dimensions \n Iteration {i} \n  (Showing 20/{n_iterations} Iterations)  "
					plt.title(txt)
					plt.show(block=False)
					if i == 0:
						plt.pause(2)
					else:
						plt.pause(1)
					plt.close()			

			if DEBUG:
				for j in range(0, self.n_particles): #getting best final result
							if ( swarm[j].evaluate() < func_obj(G_best_position)):
								G_best_position = swarm[j].position.copy()
				bests.append(G_best_position)

			if(topology == 'ALL'):
				self.all_topology(swarm)
			elif(topology == 'ADJ'):
				self.adj_topology(swarm)
			elif(topology == 'ADJ2'):
				self.adj2_topology(swarm)
			else:
				print('Error topology choose')
				exit()
			
			i+=1
			if GRAPH_ITERATION_3D:
				if(i <= 40): #more than that is very close to zero and it is not possible to visualize
					self.build_visualize_animation(swarm, x, y, z)

		if GRAPH_ITERATION_3D:
			self.visualize_animation(swarm[0].bounds, x, y, z)	

		for j in range(0, self.n_particles): #getting best final result
			if ( swarm[j].evaluate() < func_obj(G_best_position)):
				G_best_position = swarm[j].position.copy()

		best_result = func_obj( G_best_position )
		global_best_execution = best_result
		print('Result %E'%(best_result)) #printing in scientific notation


	def all_topology(self, swarm):
		
		G_best_position = swarm[0].position.copy()
		
		for j in range(1, self.n_particles):
			if ( swarm[j].evaluate() < func_obj(G_best_position)):
				G_best_position = swarm[j].position.copy()
		
		for j in range(0, self.n_particles):
			swarm[j].update_velocity(G_best_position)
			swarm[j].update_position()

		if DEBUG:
			print ("\nBest Particle:", G_best_position)	
			print("Particles:")
			for particle in swarm:
				print(particle.evaluate())

	def adj_topology(self, swarm):

		for j in range(0, self.n_particles):

			predecessor = j-1
			successor = j+1
			if(predecessor < 0): # the neighbor of position zero is the last one
				predecessor = self.n_particles-1
			if(successor > self.n_particles-1): # the neighbor of last one is the zero position
				successor = 0

			G_best_position = swarm[j].position.copy()
			if ( swarm[predecessor].evaluate() < func_obj(G_best_position)):
				G_best_position = swarm[predecessor].position.copy()
			elif( swarm[successor].evaluate() < func_obj(G_best_position)):
				G_best_position = swarm[successor].position.copy()
			
			swarm[j].update_velocity(G_best_position)
			swarm[j].update_position()

		if DEBUG:
			print("Particles:")
			for particle in swarm:
				print(particle.evaluate())

	def adj2_topology(self, swarm):
		for j in range(0, self.n_particles):

			predecessor = j-1
			predecessor2 = j-2
			successor = j+1
			successor2 = j+2
			if(predecessor < 0): # the neighbor of position zero is the last one
				predecessor = self.n_particles-1
				predecessor2 = self.n_particles-2
			elif(predecessor2 < 0):
				predecessor2 = self.n_particles-1
			if(successor > self.n_particles-1): # the neighbor of last one is the zero position
				successor = 0
				successor2 = 1
			elif(successor2 > self.n_particles-1): 
				successor2 = 0

			G_best_position = swarm[j].position.copy()
			if ( swarm[predecessor].evaluate() < func_obj(G_best_position)):
				G_best_position = swarm[predecessor].position.copy()
			elif( swarm[successor].evaluate() < func_obj(G_best_position)):
				G_best_position = swarm[successor].position.copy()
			elif( swarm[successor2].evaluate() < func_obj(G_best_position)):
				G_best_position = swarm[successor2].position.copy()
			elif( swarm[predecessor2].evaluate() < func_obj(G_best_position)):
				G_best_position = swarm[predecessor2].position.copy()
			
			swarm[j].update_velocity(G_best_position)
			swarm[j].update_position()

		if DEBUG:
			print("Particles:")
			for particle in swarm:
				print(particle.evaluate())
	
	def build_visualize_animation(self, swarm, x, y, z):

		for i in range(self.n_particles):
			x.append(swarm[i].position[0])
			y.append(swarm[i].position[1])
			z.append(swarm[i].position[2])
	
	def visualize_animation(self, bounds, x, y, z):
		df = pandas.DataFrame()
		df['x'] = x
		df['y'] = y
		df['z'] = z
		interations_column = []
		for j in range(40):
			for _ in range(self.n_particles):
				interations_column.append(j)
		df['iteration'] = interations_column
		fig = px.scatter_3d(data_frame=df, x='x', y='y', z='z', animation_frame='iteration', range_x = bounds, 
			range_y = bounds, range_z = bounds,)
		fig.update_layout( margin=dict(l=0, r=0, b=0, t=0),
			scene = dict(aspectmode='cube')
			)
		fig.update_traces(marker=dict(size=3))
		fig.show()

	def visualize_3D(self, swarm, fig):
		x = []
		y = []
		z = []
		ax = fig.add_subplot(111, projection='3d')
		for i in range(self.n_particles):
			x.append(swarm[i].position[0])
			y.append(swarm[i].position[1])
			z.append(swarm[i].position[2])
		ax.axes.set_xlim3d(swarm[0].bounds)
		ax.axes.set_ylim3d(swarm[0].bounds)
		ax.axes.set_zlim3d(swarm[0].bounds)
		ax.scatter(x, y, z, c='b', marker='.')

def calibrate():
	
	global global_best_execution, DEBUG, CODE_VERBOSE, GRAPH_SOLUTION, GRAPH_SLEEP_3D 

	CODE_VERBOSE = False
	DEBUG = False
	GRAPH_ITERATION_3D = False
	GRAPH_SLEEP_3D = False

	n_particles_list = [50]
	topology_list = ['ALL', 'ADJ2', 'ADJ']
	n_iterations_list = [50, 100, 200]
	c1_list = [0.1, 0.5, 0.7, 1]
	c2_list = [0.2, 0.5, 0.8, 2]
	w_list = [0.3, 0.5, 0.9]

	df = pandas.DataFrame()
	n_iterations_column = []
	c1_column = []
	c2_column = []
	w_column = []
	n_particles_column = []
	params_columns = []
	topology_columns = []

	#list to saving results
	list_best_execution = [] #best result for execution 

	final_list_bests_iterations = [] # best result of all repetitions
	final_mean_list_best_execution = [] #mean of best results of all repetitions 
	times_repetition = 10

	#progress estimative ...
	total_progress = len(n_iterations_list) * len(c1_list) *	len(c2_list) * len(w_list ) \
		* len(n_particles_list ) * len(topology_list ) * times_repetition
	my_bar = Bar.ShadyBar('Calibrating...', max=total_progress,  suffix='%(percent)d%%')

	for t in topology_list:
		for n in n_iterations_list:
			for c in c1_list:
				for c2 in c2_list:
					for w in w_list:
						for p in n_particles_list:
							n_iterations_column.append(t)
							topology_columns.append(t)
							c1_column.append(c)	
							c2_column.append(c2)	
							w_column.append(w)
							n_particles_column.append(p)			
							print("Execution with the current params: ")
							print(f'Number of interations: {n} C1: {c} C2:{c2} w:{w} N_Particles:{p} Topology:{t}')
							params_columns.append(f'Number of interations: {n} C1: {c} C2:{c2} w:{w} N_Particles:{p} Topology:{t}')	

							for _ in range(times_repetition): 
								print('\n\n')
								my_bar.next()
								print('\n\n')
								PSO_Real(n_iterations=n, n_particles= p, c1=c, c2=c2, W=w, num_dimensions=5, topology=t)
								list_best_execution.append(global_best_execution)
								#clean globals
								global_best_execution = 0

							final_list_bests_iterations.append(min(list_best_execution))
							final_mean_list_best_execution.append(np.mean(list_best_execution))
							
							#clean lists
							list_best_execution.clear()

	df['Topology'] = topology_columns						
	df['Params'] = params_columns
	df["Number_Iterations"] = n_iterations_column 
	df["Number_Particles"] = n_particles_column
	df["C1"] = c1_column
	df['C2'] = c2_column
	df['W'] = w_column
	df["Mean bests_Results"] = final_mean_list_best_execution
	df["Best of Iterations"] = final_list_bests_iterations
	
	df.to_csv('results_params_PSO.csv', sep=';')

if __name__ == "__main__":

	'''
	ALL = everyone 
	ADJ = each particle knows the neighbors before and after it
	ADJ2 = each particle knows the two neighbors before and the two after it
	'''
	topology = 'ALL' 
	
	Calibrate = False
	if Calibrate:
		calibrate()
	else:
		PSO_Real(n_iterations=100, n_particles= 300, c1=0.5, c2=2, W=0.3, num_dimensions=5, topology = topology)

	end = time.time() #user
	user_time = end - start 
	elapsed_time = time.process_time() - t #process

	print("="*100) 
	print("User time: %s" % user_time)
	print("Process time: %s" % elapsed_time)
	print( time.strftime("%H hours %M minutes %S seconds", time.gmtime(user_time)) ) 
	print("="*100)