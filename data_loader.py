import pandas as pd
def load_fx_data(filepath='Data/fx_data.xlsx'):
    df = pd.read_excel(filepath)  # shape: (N, 3)
    fx_data = df.to_numpy().T     # Transpose during load → final shape: (3, N)
    return fx_data