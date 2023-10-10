import pandas as pd

data = pd.read_csv('D18PMIFFPMI.csv')
data['Current'] = data['Current'] * -1
data.to_csv('D18PMIFFPMI.csv')