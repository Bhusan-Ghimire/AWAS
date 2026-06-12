import kagglehub
import pandas as pd

path = kagglehub.dataset_download("sagyamthapa/nepali-housing-price-dataset")

print("Path to dataset files:", path)

file_path = r"C:\Users\HP\.cache\kagglehub\datasets\sagyamthapa\nepali-housing-price-dataset\versions\1\2020-4-27.csv"

df = pd.read_csv(file_path)
print(df.head())
