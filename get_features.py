import pandas as pd

df = pd.read_csv('Processed_Fatality.csv')

# Features from the model
features = ['COUNTYNAME', 'MILEPTNAME', 'RELJCT1NAME', 'MONTHNAME', 'ROUTENAME', 
           'CITYNAME', 'HOURNAME', 'DAY_WEEKNAME', 'WEATHERNAME', 'RD_OWNERNAME', 
           'TYP_INTNAME', 'NHSNAME', 'SP_JURNAME', 'REL_ROADNAME', 'RUR_URBNAME', 
           'LGT_CONDNAME', 'FUNC_SYSNAME', 'RELJCT2NAME']

print('Feature unique values:')
for feature in features:
    if feature in df.columns:
        unique_vals = df[feature].dropna().unique()
        print(f'{feature}: {len(unique_vals)} unique values')
        print(f'  First 10: {list(unique_vals[:10])}')
        print()

# Also get numeric features
numeric_features = ['TWAY_ID', 'SCH_BUS', 'RAIL', 'WRK_ZONE', 'STATE', 'DAY']
print('\nNumeric features stats:')
for feature in numeric_features:
    if feature in df.columns:
        print(f'{feature}: min={df[feature].min()}, max={df[feature].max()}, unique={df[feature].nunique()}')
