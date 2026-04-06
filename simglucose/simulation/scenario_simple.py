from enum import Enum
from simglucose.simulation.scenario import Action
import numpy as np
from scipy.stats import truncnorm


class Scenario(Enum):
    NO_MEAL = "no_meal"
    SINGLE_MEAL = "single_meal"
    ONE_DAY = "one_day"
    THREE_DAY = "three_day"
    RANDOM_ONE_DAY = "random_one_day"

    def __init__(self, value):
        self._value_ = value  # handle Enum value
        self._random_meals = None
        self._random_gen = None
        self._seed = 42  # Default seed for RANDOM_ONE_DAY

    def set_random_seed(self, seed=None):
        """Set random seed for RANDOM_ONE_DAY scenario."""
        if self == Scenario.RANDOM_ONE_DAY:
            self._seed = seed
            self._random_gen = np.random.RandomState(seed)
            self._random_meals = self._generate_random_meals()

    def _generate_random_meals(self):
        """Generate random meals following scenario_gen.py pattern."""
        if self._random_gen is None:
            self._random_gen = np.random.RandomState(None)

        meals = []

        # Probability of taking each meal
        # [breakfast, snack1, lunch, snack2, dinner, snack3]
        prob = [0.95, 0.3, 0.95, 0.3, 0.95, 0.3]
        time_lb = np.array([5, 9, 10, 14, 16, 20]) * 60
        time_ub = np.array([9, 10, 14, 16, 20, 23]) * 60
        time_mu = np.array([7, 9.5, 12, 15, 18, 21.5]) * 60
        time_sigma = np.array([60, 30, 60, 30, 60, 30])
        amount_mu = [45, 10, 70, 10, 80, 10]
        amount_sigma = [10, 5, 10, 5, 10, 5]

        for p, tlb, tub, tbar, tsd, mbar, msd in zip(
            prob, time_lb, time_ub, time_mu, time_sigma, amount_mu, amount_sigma
        ):
            if self._random_gen.rand() < p:
                tmeal = np.round(
                    truncnorm.rvs(
                        a=(tlb - tbar) / tsd,
                        b=(tub - tbar) / tsd,
                        loc=tbar,
                        scale=tsd,
                        random_state=self._random_gen,
                    )
                )
                amount = max(round(self._random_gen.normal(mbar, msd)), 0)
                meals.append((int(tmeal), amount))

        return meals

    def get_action(self, t, body_weight=None):
        """
        Get meal action for the given time.

        Args:
            t: time in minutes
            body_weight: weight of the patient in kg (optional)

        Returns:
            Action namedtuple with meal amount
        """
        # Force t to int for exact time matching
        t = int(t)

        # Handle RANDOM_ONE_DAY separately
        if self == Scenario.RANDOM_ONE_DAY:
            # Generate meals at t=0 if not already generated
            if t == 0 and self._random_meals is None:
                self._random_meals = self._generate_random_meals()

            # Check if current time matches any random meal time
            if self._random_meals is not None:
                for meal_time, meal_amount in self._random_meals:
                    if meal_time == t:
                        return Action(meal=meal_amount)
            return Action(meal=0)

        carb_times_in_hour = {
            Scenario.NO_MEAL: [],
            Scenario.SINGLE_MEAL: [6],  # Assuming a meal at 6:00
            Scenario.ONE_DAY: [7, 12, 19],  # Meals at 7:00, 12:00, 19:00
            Scenario.THREE_DAY: [
                h + 24 * d for d in range(3) for h in [7, 12, 19]
            ],  # Meals at 7:00, 12:00, 19:00 for 3 days
        }

        # Predefined carb amounts for each meal time
        carb_amounts = {
            Scenario.NO_MEAL: [],
            Scenario.SINGLE_MEAL: [75],  # 75g of carbs for single meal
            Scenario.ONE_DAY: [40, 60, 70],  # 40g, 60g, 70g of carbs per meal
            Scenario.THREE_DAY: [
                40,
                50,
                70,
            ]
            * 3,  # 40g, 50g, 70g of carbs per meal for 3 days
        }
        if body_weight is not None:
            carb_amounts[Scenario.ONE_DAY] = [
                0.5 * body_weight,
                0.8 * body_weight,
                0.8 * body_weight,
            ]
            carb_amounts[Scenario.THREE_DAY] = [
                0.5 * body_weight,
                0.8 * body_weight,
                0.8 * body_weight,
            ] * 3

        # Convert hours to minutes for comparison
        carb_times_in_minutes = [h * 60 for h in carb_times_in_hour[self]]

        # Check if current time matches any meal time exactly
        if t in carb_times_in_minutes:
            idx = carb_times_in_minutes.index(t)
            return Action(meal=carb_amounts[self][idx])
        return Action(meal=0)

    def get_carb(self, t, body_weight=None):
        """
        Returns carbs only at exact meal times, 0 otherwise.
        """
        return self.get_action(t, body_weight).meal

    @property
    def max_t(self):
        """Maximum simulation time in minutes for this scenario."""
        return {
            Scenario.NO_MEAL: 1000,  # 16 hours + 40 minutes
            Scenario.SINGLE_MEAL: 1080,  # 18 hours
            Scenario.ONE_DAY: 1450,  # 24 hours + 10 minutes
            Scenario.THREE_DAY: 4330,  # 72 hours + 10 minutes
            Scenario.RANDOM_ONE_DAY: 1450,  # 24 hours + 10 minutes
        }[self]


if __name__ == "__main__":
    # Test exact meal times
    print("Testing exact meal times:")
    print(
        f"NO_MEAL at 420min (7:00): {Scenario.NO_MEAL.get_carb(420, 70)}"
    )  # Should be 0
    print(
        f"SINGLE_MEAL at 360min (6:00): {Scenario.SINGLE_MEAL.get_carb(360, 70)}"
    )  # Should be 50
    print(
        f"ONE_DAY at 420min (7:00): {Scenario.ONE_DAY.get_carb(420, 70)}"
    )  # Should be 0.5*70=35
    print(
        f"ONE_DAY at 720min (12:00): {Scenario.ONE_DAY.get_carb(720, 70)}"
    )  # Should be 0.8*70=56
    print(
        f"ONE_DAY at 1140min (19:00): {Scenario.ONE_DAY.get_carb(1140, 70)}"
    )  # Should be 0.8*70=56
    print(
        f"THREE_DAY at 420min (7:00 day 1): {Scenario.THREE_DAY.get_carb(420, 70)}"
    )  # Should be 0.5*70=35

    print("\nTesting non-meal times (should all return 0):")
    print(f"SINGLE_MEAL at 365min: {Scenario.SINGLE_MEAL.get_carb(365)}")  # Should be 0
    print(f"ONE_DAY at 425min: {Scenario.ONE_DAY.get_carb(425)}")  # Should be 0
    print(f"THREE_DAY at 425min: {Scenario.THREE_DAY.get_carb(425)}")  # Should be 0

    print("\nTesting without body_weight:")
    print(f"ONE_DAY at 420min (7:00): {Scenario.ONE_DAY.get_carb(420)}")  # Should be 40
    print(
        f"ONE_DAY at 720min (12:00): {Scenario.ONE_DAY.get_carb(720)}"
    )  # Should be 50
    print(
        f"ONE_DAY at 1140min (19:00): {Scenario.ONE_DAY.get_carb(1140)}"
    )  # Should be 70
