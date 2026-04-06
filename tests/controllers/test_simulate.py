from simglucose.controller.pid_ctrller import PIDController
from simglucose.simulation.user_interface import simulate
from simglucose.simulation.scenario_gen import RandomScenario
from datetime import datetime, timedelta

pid_controller = PIDController(
    k_P=0.001,
    k_I=0.00001,
    k_D=0.001,
    target_BG=150,
)

now = datetime.now()
start_hour = timedelta(hours=2)
start_time = datetime.combine(now.date(), datetime.min.time()) + start_hour


s = simulate(
    controller=pid_controller,
    animate=True,
    parallel=False,
    save_path="default",
    sim_time=timedelta(hours=24),
    scenario=RandomScenario(start_time=start_time, seed=6),
    patient_names=["adolescent#003"],
    cgm_name="Dexcom",
    cgm_seed=1,
    insulin_pump_name="Insulet",
)

print("done")
