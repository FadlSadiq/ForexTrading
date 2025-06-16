# main.py

import os
import argparse
import numpy as np
import pandas as pd

from data_loader import load_fx_data
from generate_fx_data import download_and_save_fx_data
from lstm_model import train_lstm_model
from fxtrading_class import FXTrading

def main(update_data: bool, horizon: int = 50):
    data_path = 'Data/fx_data.xlsx'

    # 1. 可选更新数据
    if update_data or not os.path.exists(data_path):
        print("Downloading fresh FX data from Yahoo Finance...")
        download_and_save_fx_data(filepath=data_path)
    else:
        print("Using cached dataset from", data_path)

    # 2. 加载真实数据，期望 load_fx_data 返回 shape (3, N) 的 numpy 数组
    real_fx_data = load_fx_data(filepath=data_path)
    total_len = real_fx_data.shape[1]
    print(f"Loaded real_fx_data with shape {real_fx_data.shape} (交易日数={total_len})")

    # 3. 校验 horizon
    max_sim_days = 90
    if horizon > max_sim_days:
        print(f"Warning: horizon ({horizon}) > {max_sim_days}, 强制设为 {max_sim_days}")
        horizon = max_sim_days
    if horizon <= 0:
        raise ValueError("horizon 必须为正整数")

    # 4. 动态计算训练集长度 train_len
    lookback = 30  # 与 lstm_model 中使用的 lookback 保持一致
    if total_len <= lookback:
        raise ValueError(f"数据总长度 {total_len} 不足以做 lookback {lookback} 的预测初始化")
    desired_train = 900
    if total_len >= desired_train + horizon:
        train_len = desired_train
    else:
        train_len = total_len - horizon
        print(f"数据长度 {total_len} 不足以用 {desired_train} 交易日训练 + {horizon} 天模拟，改为 train_len = {train_len} 天训练")
    if train_len < lookback:
        raise ValueError(f"训练长度 train_len={train_len} 小于 lookback={lookback}，无法进行 LSTM 训练。请提供更多历史数据或减小 horizon。")

    # 5. 划分训练集
    train_data = real_fx_data[:, :train_len]  # shape (3, train_len)
    # 模拟时会用到真实数据索引范围 [train_len, train_len + horizon - 1]

    # 6. 训练 LSTM 模型，仅使用训练集
    print(f"Training LSTM model on first {train_len} trading days...")
    # train_lstm_model signature: train_lstm_model(real_fx_data_subset, lookback=30, epochs=20)
    model, scalers, returned_lookback = train_lstm_model(train_data, lookback=lookback, epochs=20)
    # 确保 returned_lookback == lookback

    # 7. 初始化模拟环境
    initial_index = train_len  # 下一步要预测/获取真实的起始索引
    print(f"Initializing FXTrading with initial_index = {initial_index}")
    env = FXTrading(
        real_fx_rates=real_fx_data,
        model=model,
        scalers=scalers,
        lookback=lookback,
        initial_index=initial_index
    )

    # 8. 运行模拟
    if initial_index + horizon > total_len:
        # 理论上 train_len 已确保 initial_index+horizon <= total_len
        raise ValueError(f"真实数据长度 {total_len} 不足以模拟 {horizon} 天")
    print(f"Running simulation for {horizon} trading days (indexes {initial_index} to {initial_index + horizon - 1})...")
    logs = env.run_simulation(days=horizon)

    # 9. 保存日志到 Excel
    df_logs = []
    for entry in logs:
        day = entry["day"]
        predicted = entry["predicted"]
        actual = entry["actual"]
        pnl = entry["floating_pnl"]
        pos = entry["position_size"]
        cap = entry["capital"]
        # 假设三个货币对按顺序 USD/JPY, USD/EUR, USD/GBP
        row = {
            "day": day,
            "pred_USDJPY": predicted[0],
            "pred_USDEUR": predicted[1],
            "pred_USDGBP": predicted[2],
            "act_USDJPY": actual[0],
            "act_USDEUR": actual[1],
            "act_USDGBP": actual[2],
            "pnl_USDJPY": pnl[0],
            "pnl_USDEUR": pnl[1],
            "pnl_USDGBP": pnl[2],
            "pos_USDJPY": pos[0],
            "pos_USDEUR": pos[1],
            "pos_USDGBP": pos[2],
            "cap_USDJPY": cap[0],
            "cap_USDEUR": cap[1],
            "cap_USDGBP": cap[2],
        }
        df_logs.append(row)
    df_logs = pd.DataFrame(df_logs)
    out_path = f"simulation_logs_{train_len}train_{horizon}sim.xlsx"
    df_logs.to_excel(out_path, index=False)
    print(f"Simulation logs saved to {out_path}")

    # 10. 打印最终结果
    final_capitals = env.capital
    print("Final capitals:", final_capitals, "Return:", sum(final_capitals) / sum(env.initial_capital))

    return logs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FX Trading Simulation.")
    parser.add_argument("--update_data", action="store_true", help="Download fresh FX data")
    parser.add_argument("--horizon", type=int, default=50, help="Number of trading days to simulate/predict (max 90)")
    args = parser.parse_args()
    main(update_data=args.update_data, horizon=args.horizon)
