import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from data_loader import load_fx_data
from generate_fx_data import download_and_save_fx_data
from training_loop import train_model
from Plotting_R2 import plot_pred_vs_actual
from fxtrading_class import FXTrading

def main(update_data: bool, horizon: int = 90):
    data_path = 'Data/fx_data.xlsx'

    # 1. 可选更新数据
    if update_data or not os.path.exists(data_path):
        print("Downloading fresh FX data from Yahoo Finance...")
        download_and_save_fx_data(filepath=data_path)
    else:
        print("Using cached dataset from", data_path)

    # 2. 加载真实数据
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
    train_data = real_fx_data[:, :train_len]

    # 6. 训练 LSTM 模型（返回 history）
    print(f"Training model on first {train_len} trading days...")
    model, scalers, returned_lookback, history = train_model(train_data, lookback=lookback, epochs=300)

    # 可视化训练 Loss 与 Accuracy（非 GUI 场景）
    try:
        import matplotlib.pyplot as plt
        epochs = list(range(1, len(history['loss']) + 1))
        # Loss 曲线
        plt.figure()
        plt.plot(epochs, history['loss'], marker='o', linewidth=1)
        plt.title("Model Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig('training_result/training_loss.png')
        plt.close()

        # RMSE
        plt.figure()
        plt.plot(epochs, history['rmse'], linewidth=1)
        plt.title("Model Training RMSE")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.grid(True)
        plt.savefig('training_result/training_RMSE.png')
        plt.close()

        # Accuracy 曲线
        plt.figure()
        plt.plot(epochs, history['accuracy'], marker='o', linewidth=1)
        plt.title("Model Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.savefig('training_result/training_accuracy.png')
        plt.close()
    except ImportError:
        pass

    # 7. 初始化模拟环境
    initial_index = train_len
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
        raise ValueError(f"真实数据长度 {total_len} 不足以模拟 {horizon} 天")
    print(f"Running simulation for {horizon} trading days...")
    logs = env.run_simulation(days=horizon)

    # 9. 保存日志到 Excel
    df_logs = pd.DataFrame(logs)
    out_path = f"training_result/simulation_logs_{train_len}train_{horizon}sim.xlsx"
    df_logs.to_excel(out_path, index=False)
    print(f"Simulation logs saved to {out_path}")

    for fx in ["USDJPY", "USDEUR", "USDGBP"]:
        pred_col = f"pred_{fx}"
        act_col = f"act_{fx}"
        if pred_col in df_logs.columns and act_col in df_logs.columns:
            preds = df_logs[pred_col].values
            actuals = df_logs[act_col].values
            plot_pred_vs_actual(
                preds,
                actuals,
                title=f"{fx} Prediction vs Actual",
                save_path=f"training_result/pred_vs_actual_{fx}.png"
            )


    # 10. 计算并打印最终结果
    final_capitals = env.capital.copy()
    roi = sum(final_capitals) / sum(env.initial_capital)
    print("Final capitals:", final_capitals)
    print(f"Return: {roi:.4f}")

    # 11. 返回 logs, final_capitals, roi
    return logs, final_capitals, roi

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FX Trading Simulation.")
    parser.add_argument("--update_data", action="store_true", help="Download fresh FX data")
    parser.add_argument("--horizon", type=int, default=90, help="Number of trading days to simulate/predict (max 90)")
    args = parser.parse_args()
    main(update_data=args.update_data, horizon=args.horizon)
