
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SmartGridEnv(gym.Env):


    metadata = {"render_modes": ["human"]}

    # Energy constants
    BATTERY_CAPACITY = 20.0      # kWh
    BATTERY_EFFICIENCY = 0.92    # charge/discharge efficiency
    BATTERY_MAX_RATE = 5.0       # max charge/discharge rate per hour (kW)
    MAX_SOLAR = 10.0             # kW peak
    MAX_WIND = 8.0               # kW peak
    MAX_DEMAND = 12.0            # kW peak
    MAX_PRICE = 0.35             # $/kWh
    MIN_PRICE = 0.05             # $/kWh

    # Reward weights
    W_RENEWABLE = 0.4
    W_COST = 1.5
    W_BLACKOUT = 5.0
    W_SELL = 0.6
    W_BALANCE = 0.1

    def __init__(self, noise_level=0.1, render_mode=None):
        super().__init__()
        self.noise_level = noise_level
        self.render_mode = render_mode

        # State: [solar, wind, demand, battery_soc, price, sin_t, cos_t, dow]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )
        # Actions: 0=Idle, 1=Charge, 2=Discharge, 3=BuyGrid, 4=SellGrid
        self.action_space = spaces.Discrete(5)

        self.reset()

    # ------------------------------------------------------------------
    # Core generation profiles
    # ------------------------------------------------------------------

    def _solar_output(self, hour):
        """Solar generation: bell-curve peaking at noon."""
        base = np.exp(-0.5 * ((hour - 12) / 3.5) ** 2)
        base = base if 6 <= hour <= 20 else 0.0
        noise = np.random.normal(0, self.noise_level)
        return float(np.clip(base + noise, 0, 1)) * self.MAX_SOLAR

    def _wind_output(self, hour):
        """Wind generation: stronger at night, variable."""
        base = 0.5 + 0.4 * np.cos((hour - 3) * np.pi / 12)
        noise = np.random.normal(0, self.noise_level * 1.5)
        return float(np.clip(base + noise, 0.1, 1.0)) * self.MAX_WIND

    def _demand(self, hour):
        """Consumer demand: morning and evening peaks."""
        # Morning peak 7-9, evening peak 18-21
        morning = 0.6 * np.exp(-0.5 * ((hour - 8) / 1.5) ** 2)
        evening = 1.0 * np.exp(-0.5 * ((hour - 19) / 2.0) ** 2)
        base = 0.3 + morning + evening
        noise = np.random.normal(0, self.noise_level * 0.5)
        return float(np.clip(base + noise, 0.2, 1.2)) * self.MAX_DEMAND

    def _grid_price(self, hour):
        """Dynamic tariff: cheap overnight, expensive evening peak."""
        if 23 <= hour or hour <= 6:
            price = 0.08  # off-peak
        elif 7 <= hour <= 11:
            price = 0.18  # shoulder
        elif 12 <= hour <= 16:
            price = 0.22  # midday
        elif 17 <= hour <= 21:
            price = 0.32  # peak
        else:
            price = 0.14  # shoulder evening
        noise = np.random.normal(0, 0.01)
        return float(np.clip(price + noise, self.MIN_PRICE, self.MAX_PRICE))

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.hour = 0
        self.day_of_week = np.random.randint(0, 7)
        # Start battery at 40-60% SoC
        self.battery_soc = np.random.uniform(0.4, 0.6) * self.BATTERY_CAPACITY

        self.total_cost = 0.0
        self.total_renewable = 0.0
        self.total_demand = 0.0
        self.blackout_energy = 0.0
        self.episode_reward = 0.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        solar = self._solar_output(self.hour)
        wind = self._wind_output(self.hour)
        demand = self._demand(self.hour)
        price = self._grid_price(self.hour)

        renewable = solar + wind
        net = renewable - demand  # positive = surplus, negative = deficit

        reward = 0.0
        grid_buy = 0.0
        grid_sell = 0.0
        bat_delta = 0.0
        blackout = 0.0

        # --- Action execution ---
        if action == 0:  # Idle: balance with renewables only
            if net >= 0:
                reward += self.W_RENEWABLE * min(renewable, demand)
            else:
                blackout = abs(net)
                reward -= self.W_BLACKOUT * blackout

        elif action == 1:  # Charge battery
            charge_possible = min(
                self.BATTERY_MAX_RATE,
                (self.BATTERY_CAPACITY - self.battery_soc) / self.BATTERY_EFFICIENCY,
                max(net, 0)
            )
            bat_delta = charge_possible * self.BATTERY_EFFICIENCY
            self.battery_soc = min(self.BATTERY_CAPACITY, self.battery_soc + bat_delta)
            reward += self.W_RENEWABLE * charge_possible * 0.5
            if net < 0:
                blackout = abs(net)
                reward -= self.W_BLACKOUT * blackout

        elif action == 2:  # Discharge battery
            discharge_possible = min(
                self.BATTERY_MAX_RATE,
                self.battery_soc * self.BATTERY_EFFICIENCY
            )
            bat_delta = -discharge_possible
            self.battery_soc = max(0, self.battery_soc + bat_delta)
            covered = renewable + discharge_possible
            if covered >= demand:
                reward += self.W_RENEWABLE * demand * 0.3
            else:
                blackout = demand - covered
                reward -= self.W_BLACKOUT * blackout

        elif action == 3:  # Buy from grid
            needed = max(demand - renewable, 0)
            grid_buy = needed
            cost = grid_buy * price
            self.total_cost += cost
            reward -= self.W_COST * cost
            # Still reward renewable portion
            reward += self.W_RENEWABLE * min(renewable, demand) * 0.2

        elif action == 4:  # Sell to grid
            surplus = max(net, 0)
            grid_sell = surplus * 0.85  # sell at 85% of buy price
            sell_revenue = grid_sell * price
            self.total_cost -= sell_revenue
            reward += self.W_SELL * sell_revenue
            if net < 0:
                blackout = abs(net)
                reward -= self.W_BLACKOUT * blackout

        # Battery balance penalty (penalise extreme SoC)
        soc_ratio = self.battery_soc / self.BATTERY_CAPACITY
        balance_penalty = abs(soc_ratio - 0.5) * self.W_BALANCE
        reward -= balance_penalty

        # Accumulate stats
        self.total_renewable += min(renewable, demand)
        self.total_demand += demand
        self.blackout_energy += blackout

        self.episode_reward += reward
        self.hour += 1
        terminated = self.hour >= 24
        truncated = False

        obs = self._get_obs()
        info = self._get_info()
        info.update({
            "solar": solar,
            "wind": wind,
            "demand": demand,
            "price": price,
            "battery_soc": self.battery_soc,
            "grid_buy": grid_buy,
            "grid_sell": grid_sell,
            "blackout": blackout,
            "action": action,
            "step_reward": reward,
        })
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        solar = self._solar_output(self.hour) / self.MAX_SOLAR
        wind = self._wind_output(self.hour) / self.MAX_WIND
        demand = self._demand(self.hour) / self.MAX_DEMAND
        soc = self.battery_soc / self.BATTERY_CAPACITY
        price = (self._grid_price(self.hour) - self.MIN_PRICE) / (self.MAX_PRICE - self.MIN_PRICE)
        time_sin = (np.sin(2 * np.pi * self.hour / 24) + 1) / 2
        time_cos = (np.cos(2 * np.pi * self.hour / 24) + 1) / 2
        dow = self.day_of_week / 6.0
        return np.array([solar, wind, demand, soc, price, time_sin, time_cos, dow], dtype=np.float32)

    def _get_info(self):
        renewable_ratio = self.total_renewable / max(self.total_demand, 1e-6)
        return {
            "hour": self.hour,
            "total_cost": self.total_cost,
            "renewable_ratio": renewable_ratio,
            "blackout_energy": self.blackout_energy,
            "episode_reward": self.episode_reward,
            "battery_soc": self.battery_soc,
        }

    def render(self):
        if self.render_mode == "human":
            info = self._get_info()
            print(f"Hour {self.hour:02d} | SoC: {self.battery_soc:.1f} kWh | "
                  f"Cost: ${self.total_cost:.3f} | Renewable: {info['renewable_ratio']:.1%}")
