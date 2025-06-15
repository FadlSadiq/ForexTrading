import yfinance as yf
import numpy as np
import pandas as pd

def download_and_save_fx_data(filepath='Data/fx_data.xlsx'):
    # Define USD-based currency pairs to download (Yahoo Finance ticker format)
    symbols = ['USDJPY=X', 'USDCAD=X', 'USDCHF=X']

    # Download historical FX data
    fx_data = {}
    for symbol in symbols:
        data = yf.download(symbol, period="3y", interval="1d")
        fx_data[symbol] = data['Close']
        print(f"{symbol} - {len(data)} rows downloaded.")

    # Convert to numpy and align lengths
    fx_numpy_list = [fx_data[s].to_numpy() for s in symbols]

    # Align all currency pairs to the same length
    min_len = min(len(arr) for arr in fx_numpy_list)
    fx_numpy_list = [arr[-min_len:] for arr in fx_numpy_list]

    # Combine into shape (3, N) → each currency pair is one row (dimension)
    fx_combined = np.stack(fx_numpy_list, axis=0)  # shape: (3, N)
    fx_data = fx_combined.reshape(3, -1)

    # Transpose → shape becomes (N, 3), suitable for DataFrame format
    fx_data_T = fx_data.T

    # Convert to DataFrame
    df = pd.DataFrame(fx_data_T, columns=['USDJPY', 'USDCAD', 'USDCHF'])

    # Save to Excel
    df.to_excel(filepath, index=False)

    print(f"Saved transposed FX data to {filepath}")