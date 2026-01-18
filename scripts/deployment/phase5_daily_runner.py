import pandas as pd
import torch
from scripts.core import ModelLoader, FeatureEngine, RiskManager

def run_daily_production():
    # 1. DATA INGESTION (3:45 PM EST)
    # Pull the last 120 days of data to ensure we have a clean 90-day sequence
    raw_data = fetch_live_data(tickers, lookback=120)
    
    # 2. FEATURE ENGINEERING
    # Must match Phase 2A exactly (14 core features)
    features = FeatureEngine.apply(raw_data)
    
    # 3. INFERENCE
    # Load: models/checkpoints/lstm_phase2a_temp0.05_best.pth
    predictions = model.predict(features.tail(90))
    
    # 4. PORTFOLIO CONSTRUCTION (S2 Logic)
    # Long: Top-K | Short: Bottom-K (Filtered by y_pred < 0)
    target_portfolio = construct_s2_portfolio(predictions)
    
    # 5. RISK OVERLAY (Phase 3.5)
    # Apply Vol Targeting (8%) and Check Kill Switches
    final_orders = RiskManager.apply_constraints(target_portfolio)
    
    # 6. OUTPUT GENERATION
    final_orders.to_csv(f"data/orders/orders_{today_date}.csv")
    print("🚀 Production Loop Complete. Ready for T+1 Execution.")

if __name__ == "__main__":
    run_daily_production()