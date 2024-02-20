import pandas as pd
import os

# Function to load the dataset
def load_dataset(path = ''):
  dataset_path = os.path.join(path, 'train.csv')
  return pd.read_csv('train.csv')

# Load dataset. Call the function with the path that contains train.csv file.
df =pd.read_csv('train.csv')
print(df)