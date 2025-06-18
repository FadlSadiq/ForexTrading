import numpy as np
import torch

class FXTrading:
    def __init__(self, real_fx_rates: np.ndarray, model, scalers: list, lookback: int = 30, initial_index: int = 900):
        self.real_fx_rates = real_fx_rates
        self.model = model
        self.scalers = scalers
        self.lookback = lookback

        total_len = real_fx_rates.shape[1]
        if initial_index < lookback:
            raise ValueError(f"initial_index ({initial_index}) must be >= lookback ({lookback})")
        if total_len < initial_index:
            raise ValueError(f"Real data length {total_len} is less than initial_index {initial_index}")

        # Initialize historical window and simulation pointer
        self.Pre_fx_rates = real_fx_rates[:, initial_index - lookback:initial_index].copy()
        self.initial_index = initial_index
        self.day = 0

        # Initialize capital and margin for each currency pair
        self.initial_capital = np.array([1000.0, 1000.0, 1000.0], dtype=float)
        self.capital = self.initial_capital.copy()
        self.available_margin = self.capital.copy()
        self.leverage = np.array([5.0, 5.0, 5.0], dtype=float)
        self.position_size = np.zeros(3, dtype=float)
        self.entry_price = np.zeros(3, dtype=float)
        self.now_price = real_fx_rates[:, initial_index - 1].copy()
        self.floating_pnl = np.zeros(3, dtype=float)
        self.margin = 10.0

        # Risk control parameters
        self.max_daily_loss_ratio = 0.1
        self.max_drawdown_ratio = 0.3
        self.min_margin_ratio = 0.2
        self.daily_loss = np.zeros(3)
        self.max_capital = self.capital.copy()

    def predict_next(self):
        hist = self.Pre_fx_rates[:, -self.lookback:]
        input_seq = np.stack([
            self.scalers[i].transform(hist[i].reshape(-1, 1)).flatten()
            for i in range(3)
        ], axis=1)
        input_tensor = torch.tensor(input_seq[np.newaxis, :, :], dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(input_tensor).cpu().numpy().flatten()
        pred_unscaled = np.array([
            self.scalers[i].inverse_transform([[pred_scaled[i]]])[0, 0]
            for i in range(3)
        ], dtype=float)
        return pred_unscaled

    def update_real(self):
        idx = self.initial_index + self.day
        if idx >= self.real_fx_rates.shape[1]:
            raise IndexError(f"Index {idx} out of bounds")

        real_price = self.real_fx_rates[:, idx]
        self.now_price = real_price.copy()
        self.available_margin = self.capital - np.abs(self.position_size) * self.margin

        # Compute floating PnL for open positions
        self.floating_pnl = np.zeros_like(self.position_size)
        for i in range(3):
            if self.position_size[i] != 0 and real_price[i] != 0:
                self.floating_pnl[i] = (
                    self.position_size[i] * (real_price[i] - self.entry_price[i]) 
                    * self.leverage[i] * self.margin / real_price[i]
                )

        # Daily loss tracking
        daily_pnl = self.floating_pnl.copy()
        self.daily_loss += np.minimum(daily_pnl, 0)

        # Margin and drawdown checks
        for i in range(3):
            # Update max capital for drawdown calculation
            self.max_capital[i] = max(self.max_capital[i], self.capital[i])
            drawdown = (self.max_capital[i] - self.capital[i]) / (self.max_capital[i] + 1e-8)
            if drawdown > self.max_drawdown_ratio:
                print(f"💥 强制平仓！{i} 超过最大回撤")
                self.position_size[i] = 0
                self.capital[i] = max(self.capital[i], 100.0)

            margin_used = abs(self.position_size[i]) * self.margin
            margin_ratio = self.available_margin[i] / (margin_used + 1e-8)
            if margin_ratio < self.min_margin_ratio:
                print(f"⚠️ 保证金不足，强制平仓 {i}")
                self.capital[i] += self.floating_pnl[i]
                self.position_size[i] = 0

        # Extend historical window
        self.Pre_fx_rates = np.concatenate([self.Pre_fx_rates, real_price.reshape(3, 1)], axis=1)

    def decide_and_trade(self, predicted_price):
        for i in range(3):
            cur = self.now_price[i]
            pred = predicted_price[i]
            error = abs(pred - cur) / (cur + 1e-8)
            direction = np.sign(pred - cur)

            # Skip trading if daily loss limit reached
            if abs(self.daily_loss[i]) > self.initial_capital[i] * self.max_daily_loss_ratio:
                print(f"❌ {i} 单日损失已达上限，今日不交易")
                continue

            # Determine leverage and order size based on error
            if error < 0.005:
                leverage = 0
                quantity = 0
            elif error < 0.015:
                leverage = 3
                quantity = 1
            elif error < 0.03:
                leverage = 5
                quantity = 1
            else:
                leverage = 5
                quantity = 2

            self.leverage[i] = leverage

            # No trade if small error
            if quantity == 0 or leverage == 0:
                continue

            # Open new position or reverse existing one
            if self.position_size[i] == 0:
                self.position_size[i] = quantity * direction
                self.entry_price[i] = cur
            else:
                prev_dir = np.sign(self.position_size[i])
                if direction != prev_dir and direction != 0:
                    # Reverse position
                    self.capital[i] += self.floating_pnl[i]
                    print(f"🔁 {i} 趋势反转，反手交易")
                    self.position_size[i] = quantity * direction
                    self.entry_price[i] = cur
                else:
                    # Check stop-profit or stop-loss threshold
                    pnl = self.floating_pnl[i]
                    if pnl >= 10.0 or pnl <= -10.0:
                        self.capital[i] += pnl
                        self.position_size[i] = 0

    def run_simulation(self, days: int = 90):
        logs = []
        currency_names = ["USDJPY", "USDEUR", "USDGBP"]
        for day in range(days):
            self.day = day
            # 1) Predict next-day rates
            predicted = self.predict_next()
            # 2) Decide and place trades
            self.decide_and_trade(predicted)
            # 3) Update real rates, PnL, margin, and window
            self.update_real()

            # Print to terminal
            print(f"Day {day + 1}\n")
            for i, name in enumerate(currency_names):
                print(f"{name}: ")
                print(f"Pre_fx_rate: {predicted[i]:.6f}  real_fx_rates: {self.now_price[i]:.6f}")
                print(f"PnL: {self.floating_pnl[i]:.2f}  Pos: {self.position_size[i]:.2f}  Capital: {self.capital[i]:.2f}\n")
            print("-" * 50)

            # 4) Append detailed logs per currency
            log_entry = {"day": day + 1}
            for i, name in enumerate(currency_names):
                log_entry[f"pred_{name}"] = round(float(predicted[i]), 6)
                log_entry[f"act_{name}"] = round(float(self.now_price[i]), 6)
                log_entry[f"pnl_{name}"] = round(float(self.floating_pnl[i]), 6)
                log_entry[f"pos_{name}"] = int(self.position_size[i])
                log_entry[f"cap_{name}"] = round(float(self.capital[i]), 6)
            logs.append(log_entry)

        # Final settlement for open positions
        for i in range(3):
            if self.position_size[i] != 0 and self.now_price[i] != 0:
                self.capital[i] += (
                    self.position_size[i] * (self.now_price[i] - self.entry_price[i]) 
                    * self.leverage[i] * self.margin / self.now_price[i]
                )
                self.position_size[i] = 0

        print("Final Results:")
        total_return = sum(self.capital) / sum(self.initial_capital)
        for i, name in enumerate(currency_names):
            print(f"{name}: capital {self.capital[i]:.2f}")
        print(f"Rate of Return: {total_return:.4f}")

        return logs
