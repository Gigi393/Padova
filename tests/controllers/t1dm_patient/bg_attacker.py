"""
Simple Blood Glucose Attacker for CGM spoofing attacks.
"""

from datetime import datetime, timedelta


class BGAttacker:
    """
    Simple attacker that spoofs blood glucose readings.

    Ramps up to target (300 mg/dL), maintains for 30min, then ramps down.
    """

    def __init__(self, step: float = 1.8, maintain_duration: float = 30.0):
        """
        Args:
            attack_start_time: When attack starts (minutes)
            step: Step size for ramping up/down (mg/dL per minute)
            maintain_duration: Time to hold at 300 (minutes)
        """
        self.max_bg = 300.0  # Max spoofed BG
        self.step = step  # mg/dL/ minute = step / 18 mmol/L per minute
        self.maintain_duration = maintain_duration
        self.initial_bg = None
        self.attack_start = None
        self.rising_end = None
        self.maintain_end = None
        self.falling_end = None

    def start_attack(self, current_time, initial_bg):
        """Initialize attack parameters."""
        self.attack_start = current_time
        self.initial_bg = initial_bg
        rise_time = (self.max_bg - self.initial_bg) / self.step
        self.rising_end = self.attack_start + rise_time
        self.maintain_end = self.rising_end + self.maintain_duration
        self.falling_end = self.maintain_end + rise_time

    def get_spoofed_bg(self, current_time, real_bg):
        """
        Returns spoofed BG if attacking, otherwise returns real BG.
        """
        if self.attack_start is None or self.initial_bg is None:
            return real_bg

        if self.attack_start <= current_time < self.rising_end:
            # Ramping up
            delta = (current_time - self.attack_start) * self.step
            return min(self.initial_bg + delta, self.max_bg)
        elif self.rising_end <= current_time < self.maintain_end:
            # Maintaining max
            return self.max_bg
        elif self.maintain_end <= current_time < self.falling_end:
            # Ramping down
            delta = (current_time - self.maintain_end) * self.step
            return max(self.max_bg - delta, real_bg)
        else:
            # Attack finished
            return real_bg


if __name__ == "__main__":
    # Test
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta

    attacker = BGAttacker()

    t = 0
    max_t = 300
    attack_t = 30

    real_bg = 100
    bgs = []
    times = []
    while t < max_t:

        if t == attack_t:
            attacker.start_attack(t, real_bg)
        spoofed_bg = attacker.get_spoofed_bg(t, real_bg)
        bgs.append(spoofed_bg)
        times.append(t)
        t += 1
        print(f"t: {t}, real BG: {real_bg}, spoofed BG: {spoofed_bg}")

    plt.plot(times, bgs)
    plt.xlabel("Time")
    plt.ylabel("Spoofed BG")
    plt.title("Blood Glucose Spoofing Attack")
    plt.show()
