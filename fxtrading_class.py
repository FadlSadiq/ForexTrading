import numpy as np
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
        self.start = len(fx_rates[0]) - 1  # Starting point in the time series

        self.initial_capital = np.array([1000, 1000, 1000], dtype=float)  # Initial capital for 3 currencies
        self.capital = np.array([1000, 1000, 1000], dtype=float)  # Current capital
        self.available_margin = self.capital  # Available margin

        self.leverage = np.array([5, 5, 5])  # Leverage for each currency (can be changed when opening positions)
        self.position_size = np.array([0, 0, 0], dtype=float)  # Current position size (positive=long, negative=short)
        self.position_value = np.array([0, 0, 0], dtype=float)  # Current position value
        self.floating_pnl = np.array([0, 0, 0], dtype=float)  # Current floating profit and loss
        self.now_price = np.array([
            real_fx_rates[0][self.start],
            real_fx_rates[1][self.start],
            real_fx_rates[2][self.start]
        ], dtype=float)  # Current market price
        self.entry_price = np.array([0, 0, 0], dtype=float)  # Entry price of the current position
        self.margin = 10  # Margin required per position

        self.model = None  # Placeholder for custom prediction model

    def check_liquidation(self, cap_num, maintenance_margin_ratio_threshold=0.3):
        """
        Check if the position should be liquidated (trigger forced liquidation).

        Args:
            cap_num (int): Index of the currency (0, 1, 2)
            maintenance_margin_ratio_threshold (float): Liquidation threshold (default 30%)

        Returns:
            bool: True if should liquidate, False otherwise
        """
        equity = self.capital[cap_num] + self.floating_pnl[cap_num]

        if equity / (self.margin * abs(self.position_size[cap_num])) < maintenance_margin_ratio_threshold:
            return True

        return False

    def close_position(self, cap_num, close_price):
        """
        Close the position and calculate realized PnL.

        Args:
            cap_num (int): Index of the currency
            close_price (float): Current market price

        Returns:
            float: Realized PnL
        """
        return (close_price - self.entry_price[cap_num]) * self.position_size[cap_num] * self.margin * self.leverage[cap_num] / close_price

    def predict_fx_rate(self, data=None):
        lookback = 30
        current_idx = self.start + self.day

        if current_idx < lookback:
            # Not enough history to predict, copy previous rate
            last_fx = self.Pre_fx_rates[:, -1].reshape(3, 1)
            self.Pre_fx_rates = np.concatenate([self.Pre_fx_rates, last_fx], axis=1)
            return

        # Prepare input sequence from real FX data (normalized)
        input_seq = np.stack([
            self.scalers[i].transform(
            self.real_fx_rates[i][current_idx - lookback:current_idx].reshape(-1, 1)
            ).flatten()
            for i in range(3)
        ]).T  # Shape: (30, 3)

        input_tensor = torch.tensor(input_seq[np.newaxis, :, :], dtype=torch.float32)

        # Model inference
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(input_tensor).numpy().flatten()

        # Inverse scale to original FX values
        pred_unscaled = np.array([
            self.scalers[i].inverse_transform([[pred_scaled[i]]])[0, 0]
            for i in range(3)
        ])

        self.Pre_fx_rates = np.concatenate([self.Pre_fx_rates, pred_unscaled.reshape(3, 1)], axis=1)

    def open_position(self, cap_num, any): #Needs to be designed manually
        """
        Decide how to open a position.

        Args:
            cap_num (int): Index of the currency

        Returns:
            (int, float): (action, number of lots)
                action: 0 = LONG, 1 = SHORT, 2 = HOLD
                buy_num: number of lots
        """
        action = 1
        buy_num = 10

        return action, buy_num

    def decide_action(self, any): #Needs to be designed manually
        """
        Decide what to do with an existing position.

        Returns:
            (int, float): (action, number of lots)
                action: 0 = ADD, 1 = CLOSE, 2 = HOLD
        """
        action = 2
        buy_num = 1

        return action, buy_num

    def update_entry_price(self, cap_num, add_price, old_position, add_position):
        """
        Update the average entry price after adding position.

        Args:
            cap_num (int): Index of the currency
            add_price (float): Price of new added position
            old_position (float): Previous position size
            add_position (float): New added position size
        """
        old_value = abs(old_position) * self.margin * self.leverage[cap_num]
        add_value = abs(add_position) * self.margin * self.leverage[cap_num]

        self.entry_price[cap_num] = (self.entry_price[cap_num] * old_value + add_price * add_value) / (old_value + add_value)

    def update(self):
        """
        Update environment for current day (prices, margins, PnL, position value).
        """
        self.now_price = np.array([
            self.real_fx_rates[0][self.start + self.day],
            self.real_fx_rates[1][self.start + self.day],
            self.real_fx_rates[2][self.start + self.day]
        ])

        self.available_margin = self.capital - abs(self.position_size) * self.margin
        self.position_value = abs(self.margin * self.position_size * self.leverage)
        self.floating_pnl = self.position_size * (self.now_price - self.entry_price) * self.leverage * self.margin / self.now_price

    def run_days(self, max_days=None):
        """
        Run simulation over multiple days.

        Args:
            max_days (int): Number of days to run
        """
        for day in range(max_days):
            self.day = day
            self.update()
            self.predict_fx_rate(None) #

            # Print current state
            print("Day ", day + 1)
            print(" ")

            for i, name in enumerate(["USD/JPY", "USD/EUR", "USD/GBP"]):
                print(name + ":")
                print("Pre_fx_rate: ", self.Pre_fx_rates[i][day + self.start], "real_fx_rates: ", self.now_price[i])
                print("Capital: ", self.capital[i], "available_margin: ", self.available_margin[i], "position_size: ", self.position_size[i], "leverage: ", self.leverage[i])
                print("floating_pnl: ", self.floating_pnl[i], "entry_price: ", self.entry_price[i], "position_value: ", self.position_value[i])
                print(" ")

            # Main trading loop for each currency
            for cap_num in range(3):
                # Check for liquidation
                if self.position_size[cap_num] != 0 and self.check_liquidation(cap_num):
                    print("Liquidation triggered!")
                    self.capital[cap_num] += self.close_position(cap_num, self.now_price[cap_num])
                    self.position_size[cap_num] = 0

                # If no position, try to open new position
                if self.position_size[cap_num] == 0 and self.capital[cap_num] > 0:
                    action, num = self.open_position(cap_num, None)
                    if action == 0 and num * self.margin <= self.available_margin[cap_num]:
                        self.position_size[cap_num] += num
                        self.entry_price[cap_num] = self.now_price[cap_num]
                    elif action == 1 and num * self.margin <= self.available_margin[cap_num]:
                        self.entry_price[cap_num] = self.now_price[cap_num]
                        self.position_size[cap_num] -= num
                else:
                    # If position exists, decide what to do
                    action, num = self.decide_action(None)

                    if action == 0:  # ADD position
                        self.update_entry_price(cap_num, self.now_price[cap_num], self.position_size[cap_num], num)
                        self.position_size[cap_num] += num
                    elif action == 1:  # CLOSE position
                        self.capital[cap_num] += self.close_position(cap_num, self.now_price[cap_num])
                        self.position_size[cap_num] = 0

        # Final update after run
        self.capital += self.position_size * (self.now_price - self.entry_price) * self.leverage * self.margin / self.now_price
        print("Final Results:")
        print("USD/JPY:  capital", self.capital[0])
        print("USD/EUR:  capital", self.capital[1])
        print("USD/GBP:  capital", self.capital[2])
        print("Rate of Return: ", sum(self.capital) / sum(self.initial_capital))