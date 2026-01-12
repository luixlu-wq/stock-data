import pandas as pd

# Load processed data
df = pd.read_parquet('data/processed/train.parquet')

print('='*70)
print(f'DATA COLUMNS ({len(df.columns)} total)')
print('='*70)

print('\n=== RAW DATA COLUMNS (7) ===')
orig = ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']
for c in orig:
    if c in df.columns:
        print(f'  - {c}')

print('\n=== TECHNICAL INDICATORS (16) ===')
tech = [c for c in df.columns if any(x in c for x in ['SMA', 'EMA', 'RSI', 'MACD', 'BB_', 'ATR', 'OBV', 'Volume_SMA'])]
for c in sorted(tech):
    print(f'  - {c}')

print('\n=== MOMENTUM & VOLATILITY (4) ===')
mom = [c for c in df.columns if 'momentum' in c or 'volatility' in c]
for c in sorted(mom):
    print(f'  - {c}')

print('\n=== MARKET FEATURES (4) ===')
mkt = [c for c in df.columns if 'market' in c or 'vs_market' in c]
for c in sorted(mkt):
    print(f'  - {c}')

print('\n=== TIME FEATURES (6) ===')
time = [c for c in df.columns if any(x in c for x in ['day_of', 'month', 'quarter', 'is_'])]
for c in sorted(time):
    print(f'  - {c}')

print('\n=== VOLUME & PRICE FEATURES (7) ===')
other = [c for c in df.columns if any(x in c for x in ['volume_', 'price_volume', 'gap', 'high_low', 'returns'])]
for c in sorted(other):
    print(f'  - {c}')

print('\n=== TARGET VARIABLES (3) ===')
tgt = [c for c in df.columns if 'target' in c]
for c in sorted(tgt):
    print(f'  - {c}')

print('\n' + '='*70)
print('ALL COLUMNS:')
print('='*70)
for i, col in enumerate(df.columns, 1):
    print(f'{i:2d}. {col}')

print(f'\nTotal: {len(df.columns)} columns')
print(f'Shape: {df.shape}')
