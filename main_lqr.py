import numpy as np
import matplotlib.pyplot as plt
import os
import control as ctrl
from pylab import *
import casadi as ca
from casadi.tools import *
import torch

############### Seetings ######################

folder_path = "C:/Users/Xuehua Xiao/Desktop/S1_CartPole/lqr data collecting" # folder to save files

# Time settings
T = 3.25  # Total time (seconds)
dt = 0.05  # Time step (seconds)
t = np.arange(0, T, dt)

x_data_shape = (len(t)-1, 4)
u_data_shape = (len(t)-1, 1)

############### LQR ######################

# system parameters
A = np.array([
    [0, 1, 0, 0],
    [0, -0.1, 3, 0],
    [0, 0, 0, 1],
    [0, -0.5, 30, 0]
])

B = np.array([
    [0],
    [2],
    [0],
    [5]
])

C = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

D = np.array([[0],[0],[0],[0]])

# Cost matrices
Q = np.diag([10, 1, 10, 1])
R = np.array([[1]])

# Convert to CasADi matrices
A_casadi = ca.MX(A)
B_casadi = ca.MX(B)
Q_casadi = ca.MX(Q)
R_casadi = ca.MX(R)

# Riccati equation
P = ca.MX.sym('P', 4, 4)
Riccati_eq = A_casadi.T @ P + P @ A_casadi - P @ B_casadi @ ca.inv(R_casadi) @ B_casadi.T @ P + Q_casadi

# Define a function to solve the Riccati equation
Riccati_function = ca.Function('Riccati_function', [P], [Riccati_eq])

# Define an optimization problem to find P such that Riccati_eq = 0
nlp = {'x': ca.reshape(P, -1, 1), 'f': ca.norm_fro(Riccati_eq)}

# Create an NLP solver
opts = {'ipopt.print_level': 0, 'print_time': 0}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# Initial guess for P
P_guess = np.eye(A.shape[0]).reshape(-1, 1)

# Solve the Riccati equation
sol = solver(x0=P_guess)
P_sol = ca.reshape(sol['x'], A.shape[0], A.shape[1])

# Compute the LQR gain matrix K
K = np.linalg.inv(R) @ B.T @ P_sol.full()

print("Solution to CARE (P):")
print(P_sol.full())
print("\nLQR Gain Matrix (K):")
print(K)

# check closed loop response
# A_cl = A - np.dot(B, K)
# sys = ctrl.ss(A_cl, B, C, D)

# time, response = ctrl.step_response(sys)

# num_outputs = response.shape[0]

# for i in range(num_outputs):
#     plt.plot(time, response[i, 0, :], label=f'Output {i+1}')

# plt.xlabel('Time')
# plt.ylabel('Response')
# plt.legend()
# plt.show()

# ############### Control Loop ######################
# # define variables to save collected x and u data
# x_list = []
# u_list = []

# Define the initial states range
rng_x = np.linspace(-1,1,50) # 50 x_0 samples
rng_theta = np.linspace(-np.pi/4,np.pi/4,50) # 50 theta_0 samples

rng0 = []
for m in rng_x:
    for n in rng_theta:
        rng0.append([m,n])
rng0 = np.array(rng0,dtype=float)
num_datagroup = len(rng0)
num_iterations = num_datagroup

# data collecting loop

x_all_tensor = torch.zeros(num_iterations,*x_data_shape)
u_all_tensor = torch.zeros(num_iterations,*u_data_shape)

for turn in range(num_datagroup):
  
  num_turn = turn + 1
  num_turn_float = str(num_turn)

  x_0 = rng0[turn,0]
  x_0= round(x_0, 3)
  theta_0 = rng0[turn,1]
  theta_0= round(theta_0, 3)
 
  #save the initial states
  x0 = np.array([x_0 , 0, theta_0, 0])  # Initial states

 #   txtfile = 'initial states'
 #   txt_name = txtfile + " " + num_turn_float + '.txt'
 #   full_txt = os.path.join(folder_path, txt_name)
 #   np.savetxt(full_txt, x0, delimiter=",",fmt='%1.3f')

  # Convert K to CasADi MX
  K_casadi = ca.MX(K)

  # Define the state update function
  x = ca.MX.sym('x', A.shape[0])
  u = -K_casadi @ x
  xdot = A_casadi @ x + B_casadi @ u
  state_update = ca.Function('state_update', [x], [xdot,u])

  # Simulation
  # t = np.arange(0, T, dt)
  x_hist = np.zeros((len(t), len(x0)))
  x_hist[0] = x0
  u_hist = np.zeros(len(t)-1)

  for i in range(1, len(t)):
       xdot_val, u_val = state_update(x_hist[i-1])
       x_hist[i] = x_hist[i-1] + dt * xdot_val.full().flatten()
       u_hist[i-1] = u_val.full().item()

  # save the (N-1) x and u 
  x_save = x_hist[:-1,:]
  u_save = u_hist

  # convert to tensor
  x_tensor = torch.tensor(x_save)
  u_tensor = torch.tensor(u_save)
  u_tensor = u_tensor.unsqueeze(1) # size [59] -> size [59, 1]

  # save tensor in variables 
  x_all_tensor[turn] = x_tensor
  u_all_tensor[turn] = u_tensor

  # some plot results
  if turn in (0, 1125, 2466):
     plt.figure(figsize=(10, 8))
     plt.subplot(5, 1, 1)
     plt.plot(t, x_hist[:, 0])
     plt.ylabel('x (m)')
     plt.grid()
     plt.subplot(5, 1, 2)
     plt.plot(t, x_hist[:, 1])
     plt.ylabel('x_dot (m/s)')
     plt.grid()
     plt.subplot(5, 1, 3)
     plt.plot(t, x_hist[:, 2])
     plt.ylabel('theta (rad)')
     plt.grid()
     plt.subplot(5, 1, 4)
     plt.plot(t, x_hist[:, 3])
     plt.ylabel('theta_dot (rad/s)')
     plt.grid()
     plt.subplot(5, 1, 5)
     plt.plot(t[:-1], u_hist)
     plt.ylabel('u_value')
     plt.xlabel('Time (s)')
     plt.grid()
     # plt.show()

     # save plot 
     plotfile = "plt"
     plot_name = plotfile + " " + num_turn_float + '.png'
     full_plot = os.path.join(folder_path, plot_name)
     plt.savefig(full_plot)

# convert list to tensor
# x_all_tensor = torch.cat(x_list, dim=0)
# u_all_tensor = torch.cat(u_list, dim=0)
print(f'collecting states data size -- {x_all_tensor.size()}') # [50*50, 59, 4]
print(f'collecting inputs data size -- {u_all_tensor.size()}') # [50*50, 59, 1] 

# save collecting data in torch PT file
torch.save(x_all_tensor, os.path.join(folder_path, f'x-collecting.pt'))
torch.save(u_all_tensor, os.path.join(folder_path, f'u-collecting.pt'))