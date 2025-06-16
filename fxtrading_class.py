import numpy as np
import torch

class FXTrading:
    def __init__(self, real_fx_rates: np.ndarray, model, scalers: list, lookback: int = 30, initial_index: int = 900):
        """
        Initialize the FXTrading simulation environment for walk-forward prediction and trading simulation.

        Args:
            real_fx_rates (np.ndarray): Real FX rates array, shape (3, T), T >= initial_index + days_to_simulate
            model: Trained prediction model (e.g., PyTorch model), input shape=(batch, lookback, 3), output shape=(batch, 3)
            scalers (list): List of length-3 scalers, fit on training data (before initial_index)
            lookback (int): Sliding window size for prediction, e.g., 30
            initial_index (int): Index in real_fx_rates at which simulation starts (i.e., next day index for prediction)
        """
        self.real_fx_rates = real_fx_rates
        self.model = model
        self.scalers = scalers
        self.lookback = lookback

        total_len = real_fx_rates.shape[1]
        if initial_index < lookback:
            raise ValueError(f"initial_index ({initial_index}) must be >= lookback ({lookback})")
        if total_len < initial_index:
            raise ValueError(f"Real data length {total_len} is less than initial_index {initial_index}, cannot initialize")

        # Prepare initial history window: last `lookback` days before initial_index
        start_hist = initial_index - lookback
        self.Pre_fx_rates = real_fx_rates[:, start_hist:initial_index].copy()  # shape (3, lookback)

        # Simulation parameters
        self.initial_index = initial_index  # Next day index to predict and retrieve real price
        self.day = 0  # Simulation day counter: 0 to days-1; real index = initial_index + day

        # Initialize capital and positions
        self.initial_capital = np.array([1000.0, 1000.0, 1000.0], dtype=float)
        self.capital = self.initial_capital.copy()
        self.available_margin = self.capital.copy()
        self.leverage = np.array([5.0, 5.0, 5.0], dtype=float)
        self.position_size = np.zeros(3, dtype=float)
        self.entry_price = np.zeros(3, dtype=float)
        # now_price initialized to last known real price at index initial_index-1
        self.now_price = real_fx_rates[:, initial_index - 1].copy()
        self.floating_pnl = np.zeros(3, dtype=float)
        self.margin = 10.0

    def predict_next(self):
        """
        Predict next day's FX rates based on the last `lookback` days in self.Pre_fx_rates.
        Returns:
            np.ndarray shape (3,): predicted FX rates
        """
        if self.Pre_fx_rates.shape[1] < self.lookback:
            raise ValueError("Pre_fx_rates length is less than lookback, cannot predict")
        # Extract last lookback columns
        hist = self.Pre_fx_rates[:, -self.lookback:]  # shape (3, lookback)
        # Normalize and prepare input sequence of shape (1, lookback, 3)
        input_seq = np.stack([
            self.scalers[i].transform(hist[i].reshape(-1, 1)).flatten()
            for i in range(3)
        ], axis=1)  # shape (lookback, 3)
        input_tensor = torch.tensor(input_seq[np.newaxis, :, :], dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(input_tensor).cpu().numpy().flatten()
        # Inverse transform
        pred_unscaled = np.array([
            self.scalers[i].inverse_transform([[pred_scaled[i]]])[0, 0]
            for i in range(3)
        ], dtype=float)
        return pred_unscaled

    def update_real(self):
        """
        Retrieve real price for the current simulation day and update state:
        - Update now_price, available_margin, floating_pnl
        - Append real price to Pre_fx_rates history
        """
        idx = self.initial_index + self.day
        total_len = self.real_fx_rates.shape[1]
        if idx >= total_len:
            raise IndexError(f"Real price index {idx} out of bounds for length {total_len}")
        real_price = self.real_fx_rates[:, idx]
        self.now_price = real_price.copy()
        # Update margins
        self.available_margin = self.capital - np.abs(self.position_size) * self.margin
        # Compute floating PnL
        self.floating_pnl = np.zeros_like(self.position_size)
        for i in range(3):
            if self.position_size[i] != 0 and real_price[i] != 0:
                self.floating_pnl[i] = (
                    self.position_size[i]
                    * (real_price[i] - self.entry_price[i])
                    * self.leverage[i]
                    * self.margin
                    / real_price[i]
                )
        # Append real price to history
        self.Pre_fx_rates = np.concatenate([self.Pre_fx_rates, real_price.reshape(3, 1)], axis=1)
        # Optionally keep only last lookback columns:
        # if self.Pre_fx_rates.shape[1] > self.lookback:
        #     self.Pre_fx_rates = self.Pre_fx_rates[:, -self.lookback:]

    def decide_and_trade(self, predicted_price):
        """
        Execute trading decisions based on predicted_price and current state.
        Replace or extend this logic with your specific algorithm.
        Args:
            predicted_price (np.ndarray): shape (3,), predicted next-day FX rates
        """
        # Example simple threshold strategy; adjust per your algorithm
        threshold = 0.01
        for i in range(3):
            cur = self.now_price[i]
            pred = predicted_price[i]
            # No position: decide to open
            if self.position_size[i] == 0:
                if pred > cur * (1 + threshold):
                    num = 1
                    if num * self.margin <= self.available_margin[i]:
                        self.position_size[i] += num
                        self.entry_price[i] = cur
                elif pred < cur * (1 - threshold):
                    num = 1
                    if num * self.margin <= self.available_margin[i]:
                        self.position_size[i] -= num
                        self.entry_price[i] = cur
            else:
                # Has position: decide to close based on PnL thresholds
                pnl = self.floating_pnl[i]
                take_profit = 10.0
                stop_loss = -10.0
                if pnl >= take_profit or pnl <= stop_loss:
                    real = self.now_price[i]
                    if real != 0:
                        self.capital[i] += (
                            self.position_size[i]
                            * (real - self.entry_price[i])
                            * self.leverage[i]
                            * self.margin
                            / real
                        )
                    self.position_size[i] = 0
        # Extend with more complex logic as needed

    def run_simulation(self, days: int = 90):
        """
        Run simulation for a number of trading days:
        For each day:
          1. predict_next() -> predicted_price
          2. decide_and_trade(predicted_price)
          3. update_real() to get actual price and update state
          4. Print detailed multi-line info per currency
          5. Record logs
        After loop, close any open positions and print final results.

        Args:
            days (int): Number of trading days to simulate
        Returns:
            logs (list of dict): Each dict contains day, predicted, actual, floating_pnl, position_size, capital
        """
        currency_names = ["USD/JPY", "USD/EUR", "USD/GBP"]
        total_len = self.real_fx_rates.shape[1]
        if self.initial_index + days > total_len:
            raise ValueError(f"Not enough real data to simulate {days} days: need index up to {self.initial_index + days - 1}, but have {total_len}")
        logs = []
        for day in range(days):
            self.day = day
            # 1. Predict next-day price
            predicted = self.predict_next()
            # 2. Decision and trade based on predicted
            self.decide_and_trade(predicted)
            # 3. Update real next-day price
            self.update_real()
            # 4. Print info
            print(f"Day {day + 1}\n")
            real_idx = self.initial_index + day
            for i, name in enumerate(currency_names):
                pre = predicted[i]
                actual = self.real_fx_rates[i, real_idx]
                cap = self.capital[i]
                avail = self.available_margin[i]
                pos = self.position_size[i]
                lev = self.leverage[i]
                pos_value_i = abs(self.margin * pos * lev)
                pnl = self.floating_pnl[i]
                entry = self.entry_price[i]
                print(f"{name}:")
                print(f"Pre_fx_rate: {pre:.6f} real_fx_rates: {actual:.6f}")
                print(f"Capital: {cap:.2f} available_margin: {avail:.2f} position_size: {pos:.2f} leverage: {lev}")
                print(f"floating_pnl: {pnl:.2f} entry_price: {entry:.6f} position_value: {pos_value_i:.2f}\n")
            print("-" * 50)
            # Record log
            logs.append({
                "day": day + 1,
                "predicted": predicted.copy(),
                "actual": self.now_price.copy(),
                "floating_pnl": self.floating_pnl.copy(),
                "position_size": self.position_size.copy(),
                "capital": self.capital.copy()
            })
        # Close any open positions at final price
        final_price = self.now_price.copy()
        for i in range(3):
            if self.position_size[i] != 0 and final_price[i] != 0:
                self.capital[i] += (
                    self.position_size[i]
                    * (final_price[i] - self.entry_price[i])
                    * self.leverage[i]
                    * self.margin
                    / final_price[i]
                )
                self.position_size[i] = 0
        print("Final Results:")
        for i, name in enumerate(currency_names):
            print(f"{name}: capital {self.capital[i]:.2f}")
        print(f"Rate of Return: {sum(self.capital) / sum(self.initial_capital):.4f}")
        return logs
