import numpy as np
import os
from pylab import *
import casadi as ca
from casadi.tools import *


num_datagroup = 300 # number of data groups 

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

# Cost matrices
Q = np.diag([10, 1, 10, 1])
R = np.array([[1]])

# Convert to CasADi matrices
A_casadi = ca.MX(A)
B_casadi = ca.MX(B)
Q_casadi = ca.MX(Q)
R_casadi = ca.MX(R)

# Riccati
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


# Define the initial state range
rng_x = np.linspace(-1,1,20)
# random_x = np.random.choice(rng_x)
# x_0= f"{random_x:.3f}"
rng_theta = np.linspace(-np.pi/4,np.pi/4,15)
# random_theta = np.random.choice(rng_x)
# theta_0 = f"{random_theta:.3f}"

rng0 = []
for m in rng_x:
    for n in rng_theta:
        rng0.append([m,n])
rng0 = np.array(rng0,dtype=float)

# data collecting loop
for turn in range(num_datagroup):
  
  num_turn = turn + 1
  num_turn_float = str(num_turn)
  folder_path = "C:/Users/Xuehua Xiao/Desktop/S1_CartPole/lqr data collecting"

  x_0 = rng0[turn,0]
  x_0= round(x_0, 3)
  theta_0 = rng0[turn,1]
  theta_0= round(theta_0, 3)

  x0 = np.array([x_0 , 0, theta_0, 0])  # Initial states
 #   txtfile = 'initial states'
 #   txt_name = txtfile + " " + num_turn_float + '.txt'
 #   full_txt = os.path.join(folder_path, txt_name)
 #   np.savetxt(full_txt, x0, delimiter=",",fmt='%1.3f')

  # Time settings
  T = 3  # Total time (seconds)
  dt = 0.01  # Time step (seconds)

  # Convert K to CasADi MX
  K_casadi = ca.MX(K)

  # Define the state update function
  x = ca.MX.sym('x', A.shape[0])
  u = -K_casadi @ x
  xdot = A_casadi @ x + B_casadi @ u
  state_update = ca.Function('state_update', [x], [xdot,u])

  # Simulate the system
  t = np.arange(0, T, dt)
  x_hist = np.zeros((len(t), len(x0)))
  x_hist[0] = x0
  u_hist = np.zeros(len(t))

  for i in range(1, len(t)):
       xdot_val, u_val = state_update(x_hist[i-1])
       x_hist[i] = x_hist[i-1] + dt * xdot_val.full().flatten()
       u_hist[i] = u_val.full().item()

  # Save the results to CSV files
 #   cvsfile = 'u_data'
 #   cvs_name = cvsfile + " " + num_turn_float + '.csv'
 #   full_cvs = os.path.join(folder_path, cvs_name)
 #   np.savetxt(full_cvs, u_hist, delimiter=",", fmt='%1.6f')

  # Plot some results
  import matplotlib.pyplot as plt
  if turn in (0, 61, 134, 227, 295):
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
     plt.plot(t, u_hist)
     plt.ylabel('u_value')
     plt.xlabel('Time (s)')
     plt.grid()
     # plt.show()

     # save plot 
     plotfile = "plt"
     plot_name = plotfile + " " + num_turn_float + '.png'
     full_plot = os.path.join(folder_path, plot_name)
     plt.savefig(full_plot)