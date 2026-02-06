import pandas as pd

df = pd.read_parquet('training_data_combined.parquet')
print(f'Total samples: {len(df):,}')
print(f'Open: {(df["open"]==1).sum():,}')
print(f'Closed: {(df["open"]==0).sum():,}')
print(f'Columns: {list(df.columns)}')
