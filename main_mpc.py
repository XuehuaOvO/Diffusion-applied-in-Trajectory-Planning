import casadi as ca
import numpy as np
import os
import control

############### Seetings ######################

num_datagroup = 300 # number of data groups 
folder_path = "C:/Users/Xuehua Xiao/Desktop/S1_CartPole/mpc data collecting"

N = 40 # prediction horizon



############### Dynamics Define ######################

def cart_pole_dynamics(x, u):
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

    C = np.eye(4)

    D = np.zeros((4,1))

    # state space equation
    sys_continuous = control.ss(A, B, C, D)

    # sampling time
    Ts = 0.1

    # convert to discrete time dynamics
    sys_discrete = control.c2d(sys_continuous, Ts, method='zoh')

    A_d = sys_discrete.A
    B_d = sys_discrete.B
    C_d = sys_discrete.C
    D_d = sys_discrete.D

    # States
    x_pos = x[0]
    x_dot = x[1]
    theta = x[2]
    theta_dot = x[3]

    x_next = ca.vertcat(
        A_d[0,0]*x_pos + A_d[0,1]*x_dot + A_d[0,2]*theta + A_d[0,3]*theta_dot + B_d[0,0]*u,
        A_d[1,0]*x_pos + A_d[1,1]*x_dot + A_d[1,2]*theta + A_d[1,3]*theta_dot + B_d[1,0]*u,
        A_d[2,0]*x_pos + A_d[2,1]*x_dot + A_d[2,2]*theta + A_d[2,3]*theta_dot + B_d[2,0]*u,
        A_d[3,0]*x_pos + A_d[3,1]*x_dot + A_d[3,2]*theta + A_d[3,3]*theta_dot + B_d[3,0]*u,
    )
    return x_next



############# MPC Loop #####################

# mpc parameters
Q = np.diag([10, 1, 10, 1])
R = np.array([[1]])
x_ref = ca.SX.sym('x_ref', 4)

# Define the initial states range
rng_x = np.linspace(-1,1,20)
rng_theta = np.linspace(-np.pi/4,np.pi/4,15)
rng0 = []
for m in rng_x:
    for n in rng_theta:
        rng0.append([m,n])
rng0 = np.array(rng0)

# data collecting loop
for turn in range(num_datagroup):
  # casadi_Opti
  optimizer = ca.Opti()
  
  # x and u mpc prediction along N
  X_pre = optimizer.variable(4, N + 1) 
  U_pre = optimizer.variable(1, N) 

  num_turn = turn + 1
  num_turn_float = str(num_turn)

  x_0 = rng0[turn,0]
  x_0= round(x_0, 3)
  theta_0 = rng0[turn,1]
  theta_0= round(theta_0, 3)
  
  #save the initial states
  x0 = np.array([x_0, 0, theta_0, 0])  # Initial states
  txtfile = 'initial states'
  txt_name = txtfile + " " + num_turn_float + '.txt'
  full_txt = os.path.join(folder_path, txt_name)
  np.savetxt(full_txt, x0, delimiter=",",fmt='%1.3f')

  optimizer.subject_to(X_pre[:, 0] == x0)  # Initial condition

  # cost 
  cost = 0

  for k in range(N):
      x_next = cart_pole_dynamics(X_pre[:, k], U_pre[:, k])
      optimizer.subject_to(X_pre[:, k + 1] == x_next)
      cost += Q[0,0]*X_pre[0, k]**2 + Q[1,1]*X_pre[1, k]**2 + Q[2,2]*X_pre[2, k]**2 + Q[3,3]*X_pre[3, k]**2 + U_pre[:, k]**2

  optimizer.minimize(cost)
  optimizer.solver('ipopt')
  sol = optimizer.solve()

  X_sol = sol.value(X_pre)
  U_sol = sol.value(U_pre)
  
  # Save the control inputs to CSV files
  cvsfile = 'u_data'
  cvs_name = cvsfile + " " + num_turn_float + '.csv'
  full_cvs = os.path.join(folder_path, cvs_name)
  np.savetxt(full_cvs, U_sol, delimiter=",", fmt='%1.6f')

  # plot some results
  step = np.linspace(0,N,N+1)
  step_u = np.linspace(0,N-1,N)

  import matplotlib.pyplot as plt
  if turn in (0, 61, 134, 227, 295):
     plt.figure(figsize=(10, 8))

     plt.subplot(5, 1, 1)
     plt.plot(step, X_sol[0, :])
     plt.ylabel('Position (m)')
     plt.grid()

     plt.subplot(5, 1, 2)
     plt.plot(step, X_sol[1, :])
     plt.ylabel('Velocity (m/s)')
     plt.grid()

     plt.subplot(5, 1, 3)
     plt.plot(step, X_sol[2, :])
     plt.ylabel('Angle (rad)')
     plt.grid()

     plt.subplot(5, 1, 4)
     plt.plot(step, X_sol[3, :])
     plt.ylabel('Ag Velocity (rad/s)')
     plt.grid()

     plt.subplot(5, 1, 5)
     plt.plot(step_u, U_sol[:])
     plt.ylabel('Ctl Input (N)')
     plt.xlabel('Horizon')
     plt.grid()

     # save plot 
     plotfile = "plt"
     plot_name = plotfile + " " + num_turn_float + '.png'
     full_plot = os.path.join(folder_path, plot_name)
     plt.savefig(full_plot)
