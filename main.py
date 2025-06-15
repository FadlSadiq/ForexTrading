import os
import argparse
from data_loader import load_fx_data
from generate_fx_data import download_and_save_fx_data
from lstm_model import train_lstm_model
from fxtrading_class import FXTrading

def main(update_data: bool):
    data_path = 'Data/fx_data.xlsx'

    # 1. Optionally update dataset
    if update_data or not os.path.exists(data_path):
        print("Downloading fresh FX data from Yahoo Finance...")
        download_and_save_fx_data(filepath=data_path)
    else:
        print("Using cached dataset from", data_path)

    # 2. Load real FX dataset (3 × N)
    real_fx_data = load_fx_data(filepath=data_path)

    # 3. Train the LSTM model and get scalers
    model, scalers, _ = train_lstm_model(real_fx_data)

    # 4. Use the first 900 time steps as initial predicted FX data
    initial_fx_data = real_fx_data[:, :900]

    # 5. Initialize simulation environment
    env = FXTrading(fx_rates=initial_fx_data, real_fx_rates=real_fx_data)
    env.model = model
    env.scalers = scalers

    # 6. Run trading simulation
    env.run_days(max_days=50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FX Trading Simulation.")
    parser.add_argument("--update_data", action="store_true", help="Download fresh FX data")
    args = parser.parse_args()
    main(update_data=args.update_data)