# main.py

import os
import argparse
from data_loader import load_fx_data
from generate_fx_data import download_and_save_fx_data
from lstm_model import train_lstm_model
from fxtrading_class import FXTrading

def main(update_data: bool, horizon: int):
    data_path = 'Data/fx_data.xlsx'

    # Download fresh FX data if requested or if the data file does not exist
    if update_data or not os.path.exists(data_path):
        print("Downloading fresh FX data from Yahoo Finance...")
        download_and_save_fx_data(filepath=data_path)
    else:
        print("Using cached dataset from", data_path)

    # Load the real FX dataset and train the LSTM model
    real_fx_data = load_fx_data(filepath=data_path)
    model, scalers, _ = train_lstm_model(real_fx_data)
    # Use the first 900 time steps as initial predicted FX data
    initial_fx_data = real_fx_data[:, :900]

    # Initialize the trading simulation environment
    env = FXTrading(fx_rates=initial_fx_data, real_fx_rates=real_fx_data)
    env.model = model
    env.scalers = scalers

    # Run the simulation for the specified horizon and capture the logs
    logs = env.run_days(max_days=horizon)
    # Print confirmation that the simulation ran for the correct number of days
    print(f"\n>>> Simulation complete: ran {len(logs)} days, matching the {horizon} days you specified.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FX trading simulation.")
    parser.add_argument(
        "--update_data", 
        action="store_true", 
        help="Download fresh FX data"
    )
    parser.add_argument(
        "--horizon", 
        type=int, 
        default=50, 
        help="Number of days to predict and simulate"
    )
    args = parser.parse_args()
    main(update_data=args.update_data, horizon=args.horizon)
