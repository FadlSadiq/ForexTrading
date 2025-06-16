import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class FXTrading:
    def __init__(self, fx_rates, real_fx_rates):
        """
        Initialize the FXTrading simulation environment.

        Args:
            fx_rates (numpy array): Predicted FX rates (3, N)
            real_fx_rates (numpy array): Real FX rates (3, N)
        """
        self.Pre_fx_rates = fx_rates
        self.real_fx_rates = real_fx_rates
        self.day = 0
        self.start = fx_rates.shape[1] - 1  # Starting point in the time series

        self.initial_capital = np.array([1000, 1000, 1000], dtype=float)
        self.capital = self.initial_capital.copy()
        self.available_margin = self.capital.copy()

        self.leverage = np.array([5, 5, 5], dtype=float)
        self.position_size = np.zeros(3, dtype=float)
        self.position_value = np.zeros(3, dtype=float)
        self.floating_pnl = np.zeros(3, dtype=float)

        # Initialize now_price from Pre_fx_rates to avoid out-of-bounds
        self.now_price = np.array([
            self.Pre_fx_rates[0][self.start],
            self.Pre_fx_rates[1][self.start],
            self.Pre_fx_rates[2][self.start]
        ], dtype=float)
        self.entry_price = np.zeros(3, dtype=float)
        self.margin = 10  # Margin required per position

        self.model = None
        self.scalers = None

    def check_liquidation(self, cap_num, maintenance_margin_ratio_threshold=0.3):
        equity = self.capital[cap_num] + self.floating_pnl[cap_num]
        if equity / (self.margin * abs(self.position_size[cap_num])) < maintenance_margin_ratio_threshold:
            return True
        return False

    def close_position(self, cap_num, close_price):
        return (close_price - self.entry_price[cap_num]) * self.position_size[cap_num] * self.margin * self.leverage[cap_num] / close_price

    def predict_fx_rate(self, data=None):
        lookback = 30
        current_idx = self.start + self.day

        if current_idx < lookback:
            last_fx = self.Pre_fx_rates[:, -1].reshape(3, 1)
            self.Pre_fx_rates = np.concatenate([self.Pre_fx_rates, last_fx], axis=1)
            return

        input_seq = np.stack([
            self.scalers[i].transform(
                self.real_fx_rates[i][current_idx - lookback:current_idx].reshape(-1, 1)
            ).flatten()
            for i in range(3)
        ]).T

        input_tensor = torch.tensor(input_seq[np.newaxis, :, :], dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(input_tensor).numpy().flatten()

        pred_unscaled = np.array([
            self.scalers[i].inverse_transform([[pred_scaled[i]]])[0, 0]
            for i in range(3)
        ])

        self.Pre_fx_rates = np.concatenate([self.Pre_fx_rates, pred_unscaled.reshape(3, 1)], axis=1)

    def open_position(self, cap_num, any):
        action = 1
        buy_num = 10
        return action, buy_num

    def decide_action(self, any):
        action = 2
        buy_num = 1
        return action, buy_num

    def update_entry_price(self, cap_num, add_price, old_position, add_position):
        old_value = abs(old_position) * self.margin * self.leverage[cap_num]
        add_value = abs(add_position) * self.margin * self.leverage[cap_num]
        self.entry_price[cap_num] = (
            self.entry_price[cap_num] * old_value + add_price * add_value
        ) / (old_value + add_value)

    def update(self):
        """
        Update environment for current day (prices, margins, PnL, position value).
        """
        idx = self.start + self.day
        # Use Pre_fx_rates to get current price (includes history + predictions)
        self.now_price = np.array([
            self.Pre_fx_rates[0][idx],
            self.Pre_fx_rates[1][idx],
            self.Pre_fx_rates[2][idx],
        ])

        self.available_margin = self.capital - abs(self.position_size) * self.margin
        self.position_value = abs(self.margin * self.position_size * self.leverage)
        self.floating_pnl = (
            self.position_size * (self.now_price - self.entry_price)
            * self.leverage * self.margin / self.now_price
        )

    def run_days(self, max_days=None):
        logs = []
        for day in range(max_days):
            self.day = day
            self.update()
            self.predict_fx_rate(None)

            # Print current state
            print(f"Day {day+1}\n")
            for i, name in enumerate(["USD/JPY", "USD/EUR", "USD/GBP"]):
                print(f"{name}: Pre_fx_rate = {self.Pre_fx_rates[i][self.start+day]}, now_price = {self.now_price[i]}")
                print(f"Capital = {self.capital[i]}, available_margin = {self.available_margin[i]}, position_size = {self.position_size[i]}, leverage = {self.leverage[i]}")
                print(f"floating_pnl = {self.floating_pnl[i]}, entry_price = {self.entry_price[i]}, position_value = {self.position_value[i]}\n")

            # Main trading logic
            for cap_num in range(3):
                if self.position_size[cap_num] != 0 and self.check_liquidation(cap_num):
                    self.capital[cap_num] += self.close_position(cap_num, self.now_price[cap_num])
                    self.position_size[cap_num] = 0

                if self.position_size[cap_num] == 0 and self.capital[cap_num] > 0:
                    action, num = self.open_position(cap_num, None)
                    if action == 0 and num * self.margin <= self.available_margin[cap_num]:
                        self.position_size[cap_num] += num
                        self.entry_price[cap_num] = self.now_price[cap_num]
                    elif action == 1 and num * self.margin <= self.available_margin[cap_num]:
                        self.entry_price[cap_num] = self.now_price[cap_num]
                        self.position_size[cap_num] -= num
                else:
                    action, num = self.decide_action(None)
                    if action == 0:
                        self.update_entry_price(cap_num, self.now_price[cap_num], self.position_size[cap_num], num)
                        self.position_size[cap_num] += num
                    elif action == 1:
                        self.capital[cap_num] += self.close_position(cap_num, self.now_price[cap_num])
                        self.position_size[cap_num] = 0

            logs.append((
                pd.Timestamp.now(),  # or build actual date
                "N/A",  # placeholder for action summary
                float(self.floating_pnl.sum()),
                float(self.available_margin.sum())
            ))

        # Final update
        self.capital += self.position_size * (self.now_price - self.entry_price) * self.leverage * self.margin / self.now_price
        print("Final Results:")
        print(f"Capital JPY: {self.capital[0]}")
        print(f"Capital EUR: {self.capital[1]}")
        print(f"Capital GBP: {self.capital[2]}")
        print(f"Rate of Return: {sum(self.capital)/sum(self.initial_capital)}")
        return logs
