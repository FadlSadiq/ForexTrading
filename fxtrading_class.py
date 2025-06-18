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

        self.Pre_fx_rates = real_fx_rates[:, initial_index - lookback:initial_index].copy()
        self.initial_index = initial_index
        self.day = 0

        self.initial_capital = np.array([1000.0, 1000.0, 1000.0], dtype=float)
        self.capital = self.initial_capital.copy()
        self.available_margin = self.capital.copy()
        self.leverage = np.array([5.0, 5.0, 5.0], dtype=float)
        self.position_size = np.zeros(3, dtype=float)
        self.entry_price = np.zeros(3, dtype=float)
        self.now_price = real_fx_rates[:, initial_index - 1].copy()
        self.floating_pnl = np.zeros(3, dtype=float)
        self.margin = 10.0

        # 風控參數
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

        self.floating_pnl = np.zeros_like(self.position_size)
        for i in range(3):
            if self.position_size[i] != 0 and real_price[i] != 0:
                self.floating_pnl[i] = (
                    self.position_size[i] * (real_price[i] - self.entry_price[i]) * self.leverage[i] * self.margin / real_price[i]
                )

        # 每日虧損與最大回撤偵測
        daily_pnl = self.floating_pnl.copy()
        self.daily_loss += np.minimum(daily_pnl, 0)

        for i in range(3):
            self.max_capital[i] = max(self.max_capital[i], self.capital[i])
            drawdown = (self.max_capital[i] - self.capital[i]) / self.max_capital[i]
            if drawdown > self.max_drawdown_ratio:
                print(f"💥 強制平倉！{i} 超過最大回撤")
                self.position_size[i] = 0
                self.capital[i] = max(self.capital[i], 100.0)

            margin_used = abs(self.position_size[i]) * self.margin
            margin_ratio = self.available_margin[i] / (margin_used + 1e-8)
            if margin_ratio < self.min_margin_ratio:
                print(f"⚠️ 保證金不足，強制平倉 {i}")
                self.capital[i] += self.floating_pnl[i]
                self.position_size[i] = 0

        self.Pre_fx_rates = np.concatenate([self.Pre_fx_rates, real_price.reshape(3, 1)], axis=1)

    def decide_and_trade(self, predicted_price):
        for i in range(3):
            cur = self.now_price[i]
            pred = predicted_price[i]
            error = abs(pred - cur) / cur
            direction = np.sign(pred - cur)

            if abs(self.daily_loss[i]) > self.initial_capital[i] * self.max_daily_loss_ratio:
                print(f"❌ {i} 單日損失已達上限，今日不交易")
                continue

            # 設定槓桿與建倉數量
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

            if quantity == 0 or leverage == 0:
                continue

            # 無部位 → 新建倉
            if self.position_size[i] == 0:
                self.position_size[i] = quantity * direction
                self.entry_price[i] = cur
            else:
                prev_dir = np.sign(self.position_size[i])
                if direction != prev_dir and direction != 0:
                    # 反手：先平倉再反向建倉
                    self.capital[i] += self.floating_pnl[i]
                    print(f"🔁 {i} 趨勢反轉，反手交易")
                    self.position_size[i] = quantity * direction
                    self.entry_price[i] = cur
                else:
                    pnl = self.floating_pnl[i]
                    if pnl >= 10.0 or pnl <= -10.0:
                        self.capital[i] += pnl
                        self.position_size[i] = 0

    def run_simulation(self, days: int = 90):
        logs = []
        currency_names = ["USD/JPY", "USD/EUR", "USD/GBP"]
        for day in range(days):
            self.day = day
            predicted = self.predict_next()
            self.decide_and_trade(predicted)
            self.update_real()

            print(f"Day {day + 1}\n")
            for i, name in enumerate(currency_names):
                print(f"{name}:")
                print(f"Pre_fx_rate: {predicted[i]:.6f} real_fx_rates: {self.now_price[i]:.6f}")
                print(f"Capital: {self.capital[i]:.2f} Margin: {self.available_margin[i]:.2f} Pos: {self.position_size[i]:.2f}")
                print(f"PnL: {self.floating_pnl[i]:.2f} Entry: {self.entry_price[i]:.6f}\n")
            print("-" * 50)

            logs.append({
                "day": day + 1,
                "predicted": predicted.copy(),
                "actual": self.now_price.copy(),
                "floating_pnl": self.floating_pnl.copy(),
                "position_size": self.position_size.copy(),
                "capital": self.capital.copy()
            })

        # 結算未平倉部位
        for i in range(3):
            if self.position_size[i] != 0 and self.now_price[i] != 0:
                self.capital[i] += (
                    self.position_size[i] * (self.now_price[i] - self.entry_price[i]) * self.leverage[i] * self.margin / self.now_price[i]
                )
                self.position_size[i] = 0

        print("Final Results:")
        for i, name in enumerate(currency_names):
            print(f"{name}: capital {self.capital[i]:.2f}")
        print(f"Rate of Return: {sum(self.capital) / sum(self.initial_capital):.4f}")
        return logs
