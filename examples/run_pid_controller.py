from simglucose.controller.pid_ctrller import PIDController
from simglucose.simulation.user_interface import simulate


pid_controller = PIDController(k_P=0.001, k_I=0.00001, k_D=0.001, target_BG=140)
s = simulate(controller=pid_controller)
