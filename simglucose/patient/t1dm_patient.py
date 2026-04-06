import numpy as np
import logging
import json

from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
from collections import namedtuple
from scipy.integrate import ode
from simglucose.patient.base import Patient

logger = logging.getLogger(__name__)


# Controller selection: 'pid' or 'bb' (basal-bolus)
CONTROLLER_TYPE = "pid"  # Change this to 'bb' to use BBController

Action = namedtuple("patient_action", ["CHO", "insulin"])
Observation = namedtuple("observation", ["Gsub"])

class T1DMPatient(Patient):
    SAMPLE_TIME = 1  # min
    EAT_RATE = 5  # g/min CHO
    CURRENT_FOLDER = Path(__file__).parent
    PATIENT_FOLDER = CURRENT_FOLDER / "patient_jsons"

    def __init__(
        self,
        params,
        init_state=None,
        init_bg=None,
    ):
        """
        T1DMPatient constructor.
        Inputs:
            - params: patient parameters loaded from JSON files in patient_jsons/ directory.
              Contains physiological constants (BW, Gb, Ib, etc.), model parameters (kabs, kmax, etc.),
              and initial conditions (x0, xeq). See patient_jsons/*.json for complete parameter structure.
            - init_state: customized initial state.
              If not specified, load the default initial state in
              params.x0
            - init_bg: customized initial blood glucose.
              If not specified, will use self._params.Gb * self._params.Vg
              where Gb is plasma glucose concentration (mg/dL) and Vg is glucose distribution volume (dL/kg)
            - t0: simulation start time, it is 0 by default
        """
        self._params = params
        self._init_state = init_state
        self._init_bg = init_bg
        self.t0 = 0
        self.t_start = datetime.now()
        self.reset()

    @classmethod
    def withID(cls, patient_id, **kwargs):
        """
        Construct patient by patient_id
        id are integers from 1 to 30.
        1  - 10: adolescent#001 - adolescent#010
        11 - 20: adult#001 - adult#001
        21 - 30: child#001 - child#010
        """
        groups = ["adolescent", "adult", "child"]
        group_id = (patient_id - 1) // 10
        patient_idx = patient_id - 10 * group_id
        patient_name = f"{groups[group_id]}#{patient_idx:03d}"
        return cls.withName(patient_name, **kwargs)

    @classmethod
    def withName(cls, name, **kwargs):
        """
        Construct patient by name.
        Names can be
            adolescent#001 - adolescent#010
            adult#001 - adult#001
            child#001 - child#010
        """
        file_name = f"{name}.json"
        file_path = T1DMPatient.PATIENT_FOLDER / file_name
        with file_path.open() as f:
            data = json.load(f)
        params = namedtuple("Params", data.keys())(*data.values())
        return cls(params, **kwargs)

    @property
    def basal(self):
        """
        Return the basal rate in U/min
        """
        return self._basal(self._params.u2ss, self._params.BW)

    @staticmethod
    def _basal(u2ss, BW):
        """
        Return the basal rate in U/min
        u2ss: basal rate in U/min
        BW: body weight in kg
        """
        return u2ss * BW / 6000

    @property
    def body_weight(self):
        return self._params.BW

    @property
    def carb_ratio(self):
        return self._params.CR

    @property
    def state(self):
        return self._odesolver.y

    @property
    def t(self):
        """
        Return the current time in minutes
        """
        return timedelta(minutes=self.t_elapsed) + self.t_start

    @property
    def t_elapsed(self):
        """
        Return the elapsed time in minutes since the start of the simulation
        """
        return self._odesolver.t

    @property
    def sample_time(self):
        return self.SAMPLE_TIME

    @property
    def observation(self):
        """
        return the observation from patient
        for now, only the subcutaneous glucose level is returned
        """
        GM = self.state[12]  # subcutaneous glucose (mg/kg)
        Gsub = GM / self._params.Vg  # (matlab) corrected
        observation = Observation(Gsub=Gsub)
        return observation

    def get_iob(self, include_plasma=True, subtract_baseline=False):
        """
        Calculate Insulin on Board (IOB) from patient model state variables.

        Args:
            include_plasma: If True, includes plasma insulin in IOB calculation
            subtract_baseline: If True, subtracts steady-state baseline basal IOB
                             to make it comparable with OpenAPS IOB (which only counts
                             insulin above baseline basal rate)

        Returns:
            float: IOB in Units (U)

        Notes:
            The patient model tracks insulin in different compartments (in pmol/kg):
            - state[10]: Subcutaneous insulin compartment 1
            - state[11]: Subcutaneous insulin compartment 2
            - state[5]: Plasma insulin (optional)

            IOB calculation:
            IOB (U) = (Isc1 + Isc2 [+ Ip]) * BW / 6000

            where BW is body weight in kg, and 6000 is the pmol to U conversion factor.

            When subtract_baseline=True:
            Uses the steady-state insulin (x0) as baseline, which represents the
            insulin present from continuous basal delivery at equilibrium.
        """
        try:
            # Subcutaneous insulin compartments (pmol/kg)
            isc1 = self.state[10]
            isc2 = self.state[11]

            iob_pmol_kg = isc1 + isc2

            # Optionally include plasma insulin
            if include_plasma:
                ip = self.state[5]
                iob_pmol_kg += ip

            # Subtract baseline if requested (to match OpenAPS IOB)
            if subtract_baseline:
                # Get baseline from initial steady state (x0)
                baseline_isc1 = self._params.x0[10]
                baseline_isc2 = self._params.x0[11]
                baseline_pmol_kg = baseline_isc1 + baseline_isc2

                if include_plasma:
                    baseline_pmol_kg += self._params.x0[5]

                # Subtract baseline to get IOB above baseline
                iob_pmol_kg = iob_pmol_kg - baseline_pmol_kg

            # Convert to Units: (pmol/kg) * kg / (pmol/U) = U
            iob_units = iob_pmol_kg * self.body_weight / 6000.0

            return iob_units
        except Exception as e:
            logger.error(f"Error calculating patient model IOB: {str(e)}")
            return None

    def _announce_meal(self, meal):
        """
        patient announces meal.
        The announced meal will be added to self.planned_meal
        The meal is consumed in self.EAT_RATE
        The function will return the amount to eat at current time
        """
        self.planned_meal += meal
        if self.planned_meal > 0:
            to_eat = min(self.EAT_RATE, self.planned_meal)
            self.planned_meal -= to_eat
            self.planned_meal = max(0, self.planned_meal)
        else:
            to_eat = 0
        return to_eat

    def reset(self):
        """
        Reset the patient state to default intial state
        """
        self.t_start = datetime.now()

        if self._init_state is None:
            self.init_state = np.copy(self._params.x0)
        else:
            self.init_state = self._init_state

        # follow run_simulation.m post processing code
        if self._init_bg is None:
            # (matlab)
            # Quest.fastingBG = struttura.Gb;
            # sc.BGinit = Quest.fastingBG;
            self._init_bg = self._params.Gb * self._params.Vg
            self._params.x0[12] = self.init_state[12] = self._init_bg
        else:
            # (matlab) Gpop = sc.BGinit * struttura.Vg; Gpop -> self.init_state[12]
            if self._init_bg < self._params.Gb:
                fGp = np.log(self._init_bg) ** self._params.r1 - self._params.r2
                risk = 10 * fGp**2
            else:
                risk = 0

            if self._init_bg * self._params.Vg > self._params.ke2:
                Et = self._params.ke1 * (
                    self._init_bg * self._params.Vg - self._params.ke2
                )
            else:
                Et = 0

            Gpop = self._init_bg * self._params.Vg
            GGta = -self._params.k2 - (
                self._params.Vmx
                * self._params.k2
                / self._params.kp3
                * (1 + self._params.r3 * risk)
            )
            GGtb = (
                self._params.k1 * Gpop
                - self._params.k2 * self._params.Km0
                - self._params.Vm0
                + self._params.Vmx * (1 + self._params.r3 * risk) * self._params.Ib
                + (
                    (
                        self._params.Vmx
                        * Gpop
                        * (1 + self._params.r3 * risk)
                        * (self._params.k1 + self._params.kp2)
                    )
                    - (
                        self._params.Vmx
                        * self._params.kp1
                        * (1 + self._params.r3 * risk)
                    )
                    + (
                        self._params.Vmx
                        * (self._params.Fsnc + Et)
                        * (1 + self._params.r3 * risk)
                    )
                )
                / self._params.kp3
            )
            GGtc = self._params.k1 * Gpop * self._params.Km0
            Gtop = (-GGtb - np.sqrt(GGtb**2 - 4 * GGta * GGtc)) / (2 * GGta)
            Idop = max(
                0,
                (
                    -(self._params.k1 + self._params.kp2) * Gpop
                    + self._params.k2 * Gtop
                    + self._params.kp1
                    - (self._params.Fsnc + Et)
                )
                / self._params.kp3,
            )
            Ipop = Idop * self._params.Vi
            ILop = self._params.m2 * Ipop / (self._params.m1 + self._params.m30)
            Xop = Ipop / self._params.Vi - self._params.Ib
            isc1op = max(
                0,
                ((self._params.m2 + self._params.m4) * Ipop - self._params.m1 * ILop)
                / (self._params.ka1 + self._params.kd),
            )
            isc2op = self._params.kd * isc1op / self._params.ka2
            u2op = (self._params.ka1 + self._params.kd) * isc1op
            self._params.x0[0] = self.init_state[0] = 0
            self._params.x0[1] = self.init_state[1] = 0
            self._params.x0[2] = self.init_state[2] = 0
            self._params.x0[3] = self.init_state[3] = Gpop
            self._params.x0[4] = self.init_state[4] = Gtop
            self._params.x0[5] = self.init_state[5] = Ipop
            self._params.x0[6] = self.init_state[6] = Xop
            self._params.x0[7] = self.init_state[7] = Idop
            self._params.x0[8] = self.init_state[8] = Idop
            self._params.x0[9] = self.init_state[9] = ILop
            self._params.x0[10] = self.init_state[10] = isc1op
            self._params.x0[11] = self.init_state[11] = isc2op
            self._params.x0[12] = self.init_state[12] = Gpop
            self._params.x0[13] = self.init_state[13] = self._params.Gnb
            self._params.x0[14] = self.init_state[14] = 0
            self._params.x0[15] = self.init_state[15] = (
                self._params.k01g * self._params.Gnb
            )
            self._params.x0[16] = self.init_state[16] = 0
            self._params.x0[17] = self.init_state[17] = 0
            self._params.x0[12] = self.init_state[12]

        self._last_Qsto = self.init_state[0] + self.init_state[1]
        self._last_foodtaken = 0
        self.name = self._params.names

        self._odesolver = ode(self.model).set_integrator("dopri5")
        self._odesolver.set_initial_value(self.init_state, self.t0)

        self._last_action = Action(CHO=0, insulin=0)
        self.is_eating = False
        self.planned_meal = 0

    def step(self, action):
        # Convert announcing meal to the meal amount to eat at the moment
        to_eat = self._announce_meal(action.CHO)
        action = action._replace(CHO=to_eat)

        # Detect eating or not and update last digestion amount
        if action.CHO > 0 and self._last_action.CHO <= 0:
            logger.info("t = {}, patient starts eating ...".format(self.t_elapsed))
            self._last_Qsto = self.state[0] + self.state[1]  # unit: mg
            self._last_foodtaken = 0  # unit: g
            self.is_eating = True

        if to_eat > 0:
            logger.debug("t = {}, patient eats {} g".format(self.t_elapsed, action.CHO))

        if self.is_eating:
            self._last_foodtaken += action.CHO  # g

        # Detect eating ended
        if action.CHO <= 0 and self._last_action.CHO > 0:
            logger.info("t = {}, Patient finishes eating!".format(self.t_elapsed))
            self.is_eating = False

        # Update last input
        self._last_action = action

        # ODE solver
        self._odesolver.set_f_params(
            action, self._params, self._last_Qsto, self._last_foodtaken
        )
        if self._odesolver.successful():
            self._odesolver.integrate(self._odesolver.t + self.sample_time)
        else:
            logger.error("ODE solver failed!!")
            raise

    @staticmethod
    def u_to_pmol(U):
        return U * 6000

    @staticmethod
    def pmol_to_u(pmol):
        return pmol / 6000

    @staticmethod
    def model(t, x, action, params, last_Qsto, last_foodtaken):
        dxdt = np.zeros(18)
        d = action.CHO * 1000  # g -> mg
        insulin = action.insulin * 6000 / params.BW  # U/min -> pmol/kg/min
        basal = T1DMPatient._basal(params.u2ss, params.BW)

        # Glucose in the stomach
        qsto = x[0] + x[1]
        # NOTE: Dbar is in unit mg, hence last_foodtaken needs to be converted
        # from mg to g. See https://github.com/jxx123/simglucose/issues/41 for
        # details.
        Dbar = last_Qsto + last_foodtaken * 1000  # unit: mg # (matlab) dosekempt

        # Stomach solid
        dxdt[0] = -params.kmax * x[0] + d

        if Dbar > 0:
            aa = 5 / (2 * Dbar * (1 - params.b))
            cc = 5 / (2 * Dbar * params.d)
            kgut = params.kmin + (params.kmax - params.kmin) / 2 * (
                np.tanh(aa * (qsto - params.b * Dbar))
                - np.tanh(cc * (qsto - params.d * Dbar))
                + 2
            )
            threshold = 30000  # (matlab) add threshold to correct kgut
            if Dbar <= threshold:
                kgut *= threshold / Dbar
        else:
            kgut = params.kmax

        # stomach liquid
        dxdt[1] = params.kmax * x[0] - x[1] * kgut

        # intestine
        dxdt[2] = kgut * x[1] - params.kabs * x[2]

        # Rate of appearance
        Rat = params.f * params.kabs * x[2] / params.BW
        # Glucose Production # add kcounter
        EGPt = (
            params.kp1 - params.kp2 * x[3] - params.kp3 * x[8] + params.kcounter * x[14]
        )
        # Glucose Utilization
        Uiit = params.Fsnc

        # renal excretion
        if x[3] > params.ke2:
            Et = params.ke1 * (x[3] - params.ke2)
        else:
            Et = 0

        # glucose kinetics
        # plus dextrose IV injection input u[2] if needed
        # (matlab) ignore u(5) as it is for insulin IV injection
        dxdt[3] = max(EGPt, 0) + Rat - Uiit - Et - params.k1 * x[3] + params.k2 * x[4]
        # correct Vmt with threshold
        if (x[3] / params.Vg) >= params.Gb:
            Vmt = params.Vm0 + params.Vmx * x[6]
        else:
            threshold = 60
            if (x[3] / params.Vg) > threshold:
                fGp = np.log(x[3] / params.Vg) ** params.r1 - params.r2
            else:
                fGp = np.log(threshold) ** params.r1 - params.r2
            risk = 10 * fGp**2
            Vmt = params.Vm0 + params.Vmx * x[6] * (1 + params.r3 * risk)
        Kmt = params.Km0
        Uidt = Vmt * x[4] / (Kmt + x[4])
        dxdt[4] = -Uidt + params.k1 * x[3] - params.k2 * x[4]

        # insulin kinetics
        dxdt[5] = (
            -(params.m2 + params.m4) * x[5]
            + params.m1 * x[9]
            + params.ka1 * x[10]
            + params.ka2 * x[11]
        )  # plus insulin IV injection u[3] if needed # (matlab) ignore u(4) as it is for insulin IV injection
        It = x[5] / params.Vi

        # insulin action on glucose utilization
        dxdt[6] = -params.p2u * x[6] + params.p2u * (It - params.Ib)

        # insulin action on production
        dxdt[7] = -params.ki * (x[7] - It)

        dxdt[8] = -params.ki * (x[8] - x[7])

        # insulin in the liver (pmol/kg)
        dxdt[9] = -(params.m1 + params.m30) * x[9] + params.m2 * x[5]

        # subcutaneous insulin kinetics
        dxdt[10] = insulin - (params.ka1 + params.kd) * x[10]

        dxdt[11] = params.kd * x[10] - params.ka2 * x[11]

        # subcutaneous glucose
        dxdt[12] = -params.ksc * x[12] + params.ksc * x[3]

        GSRb = params.k01g * params.Gnb
        GSRs = max(
            0,
            params.kGSRs
            * (params.Gth - (x[3] / params.Vg))
            / (max(0, It - params.Ith) + 1)
            + GSRb,
        )
        GSRd = max(0, -params.kGSRd * dxdt[3] / params.Vg)
        GSR = GSRd + max(0, x[15])

        dxdt[13] = (
            -params.k01g * x[13]
            + GSR
            + 10**9 * params.SQgluc_k2 * x[17] / params.SQgluc_Vgcn
        )
        dxdt[14] = -params.kXGn * x[14] + params.kXGn * max(0, x[13] - params.Gnb)
        #  sys(imag(sys) ~= 0) = 0;
        dxdt[15] = -params.alfaG * (x[15] - GSRs)
        # (matlab) ignore u(7) as glucagon
        dxdt[16] = -(params.SQgluc_k1 + params.SQgluc_kc1) * x[17]
        dxdt[17] = params.SQgluc_k1 * x[16] - params.SQgluc_k2 * x[17]
        dxdt = np.nan_to_num(dxdt, nan=0.0)
        dxdt[np.imag(dxdt) != 0] = 0

        # (matlab) add correction for < 0 cases
        if x[3] < 0:
            dxdt[:18] = 0

        if action.insulin > basal:
            logger.debug("t = {}, injecting insulin: {}".format(t, action.insulin))
        return dxdt


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    CtrlObservation = namedtuple("CtrlObservation", ["CGM"])

    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    patient_name = "adolescent#003"
    p = T1DMPatient.withName(patient_name)
    basal = p.basal
    t = []
    CHO = []
    insulin = []
    BG = []
    new_state = None

    # Select controller based on CONTROLLER_TYPE
    current_sim_time = datetime.now()  # Starting time for simulation
    while p.t_elapsed < 300:
        ins = basal
        carb = 0

        if p.t_elapsed == 100:
            carb = 80
        elif p.t_elapsed == 200:
            carb = 50

        if p.t_elapsed > 100 and p.t_elapsed <= 150:
            ins = p.basal * 10
        elif p.t_elapsed > 200 and p.t_elapsed <= 250:
            ins = p.basal * 5

        ctrl_obs = CtrlObservation(p.observation.Gsub)
        act = Action(insulin=ins, CHO=carb)

        t.append(p.t_elapsed)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)
        print(
            f"Time: {p.t}, time elapsed :{p.t_elapsed}, "
            f"BG: {p.observation.Gsub}, CHO: {act.CHO}, Insulin: {act.insulin}"
        )

    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(t, BG)
    ax[1].plot(t, CHO)
    ax[2].plot(t, insulin)
    plt.show()
