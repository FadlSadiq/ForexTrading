# Pulse FX 

Welcome to your intelligent foreign exchange trading playground, where deep learning meets financial strategy.

This project is a realistic, AI-driven FX trading simulator designed for both enthusiasts and researchers. It integrates a custom-built neural forecasting model and a backtesting engine, all wrapped in a sleek, interactive GUI powered by Flet.

## Model Structure
The core of the simulator is a hybrid deep learning model that combines:

- Convolutional Layers (CNN) — for extracting local time-series features
- Bidirectional GRU — for modeling sequential market dynamics
- Attention Mechanism — to focus on the most relevant time periods

We also prepare LSTM Model for comparison.

## Repository Structure
```
ForexTrading/
│
├── Assets/                             # Photo and logo assets
│   ├── image_1.png                     # Background
│   └── logo.png                        # Logo
|
├── Data/                               # Raw or preprocessed datasets
│   ├── fx_data.xlsx                    # Initial predicted data
│   └── Real_fx_data.xlsx               # Full ground truth data for simulation
│
|__ Training_Result/
│   ├── pred_vs_actual_USDEUR.png       
│   ├── pred_vs_actual_USDGBP.png       
│   ├── pred_vs_actual_USDJPY.png       
│   └── etc (other Training Result)     
|
├── data_loader.py                      # Responsible for downloading & returning FX data
├── fx_tcn_attn_model.py                # Hyrbid model definition and function
├── fxtrading_class.py                  # FXTrading class and simulation logic
├── generate_fx_data.py                 # Download and preprocess FX data
├── GUI.py                              # Flet GUI implementation
├── lstm_model.py                       # LSTM model definition and training function
├── main.py                             # Connecting between GUI and model
├── Plotting_R2.py                      # Plot R² and r
├── requirement.txt                     # Package used in the program
└── training_loop.py                    # Training the selected model
```

## Setup Instructions

1. **Clone this Repository**  
```bash
git clone https://github.com/FadlSadiq/ForexTrading
cd ForexTrading
```

2. **Install Dependencies**  
```bash
pip install -r requirements.txt
```

3. **Run Simulation**  
```bash
python GUI.py # Running the UI
python main.py # Running the model without UI
```
There are 2 models available in here, Hybrid and LSTM Model. The model being used is Hybrid Model.
For Comparison, can accessed the training_loop.py, and changed "FXCNNBiGRUAttn" to "FXLSTM" in the model