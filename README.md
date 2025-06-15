# ForexTrading Simulator 

This project simulates multi-currency Forex trading using AI-based exchange rate forecasting.


## Repository Structure
```
ForexTrading/
│
├── Data/                         # Raw or preprocessed datasets
│   ├── fx_data.xlsx              # Initial predicted data
│   └── Real_fx_data.xlsx         # Full ground truth data for simulation
│
├── data_loader.py               # Responsible for downloading & returning FX data
├── lstm_model.py                # LSTM model definition and training function
├── fxtrading_class.py           # FXTrading class and simulation logic
├── main.py                      # Main entry point: calls everything above
├── requirement.txt              # Package used in the program
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
python main.py
```