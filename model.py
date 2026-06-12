import kagglehub
import pandas as pd
import pandas as pd
import re

path = kagglehub.dataset_download("sagyamthapa/nepali-housing-price-dataset")

print("Path to dataset files:", path)

file_path = r"C:\Users\HP\.cache\kagglehub\datasets\sagyamthapa\nepali-housing-price-dataset\versions\1\2020-4-27.csv"

df = pd.read_csv(file_path)





def convert_area_to_sqft(area_str):
    # 1. Handle missing values/NaNs immediately
    if pd.isna(area_str) or str(area_str).strip() in ['N/A', 'NA', 'N/A Aana', 'N/A Sq. Feet']:
        return None
    
    # Standardize string format: lowercase and drop extra whitespaces
    area_str = str(area_str).lower().strip()
    
    try:
        # CASE 1: Hyphenated Ropani-Aana-Paisa-Daam format (e.g., "1-0-0-0 aana", "0-12-3-1")
        if '-' in area_str:
            # Extract just the hyphenated digits block (e.g., "0-12-3-1")
            clean_str = re.findall(r'[\d\.]+(?:-[\d\.]+)+', area_str)[0] #remove unit of area, and keep only the numbers in string format seperated by hyphen. 
            parts = [float(x) for x in clean_str.split('-')] #make a list of numbers in the numbers present in clear_str sting. 
            
            # If 4 parts: Ropani-Aana-Paisa-Daam
            if len(parts) == 4:
                r, a, p, d = parts
                return (r * 5476) + (a * 342.25) + (p * 85.56) + (d * 21.39)
            # If 3 parts: Ropani-Aana-Paisa
            elif len(parts) == 3:
                r, a, p = parts
                return (r * 5476) + (a * 342.25) + (p * 85.56)
        
        # CASE 2: Dot-separated Ropani-Aana-Paisa-Daam
        # Examples:
        # 3.1.0 Aana
        # 0.4.0.0 Aana
        dot_match = re.fullmatch(
            r'(\d+(?:\.\d+){2,3})\s*aana',
            area_str
        )

        if dot_match:
            parts = [float(x) for x in dot_match.group(1).split('.')]

            if len(parts) == 4:
                r, a, p, d = parts
                return (
                    r * 5476 +
                    a * 342.25 +
                    p * 85.56 +
                    d * 21.39
                )

            elif len(parts) == 3:
                r, a, p = parts
                return (
                    r * 5476 +
                    a * 342.25 +
                    p * 85.56
                )
            
        # --------------------------------------------------
        # CASE 3: Dimensions in Haat
        # Examples:
        # 20*24 Haat
        # 25/60 Haat
        # 19x40 Haat
        # --------------------------------------------------
        haat_match = re.search(
            r'(\d+(?:\.\d+)?)\s*[*x/]\s*(\d+(?:\.\d+)?)\s*haat',
            area_str
        )

        if haat_match:
            length = float(haat_match.group(1))
            width = float(haat_match.group(2))

            # 1 Haat ≈ 1.5 ft
            # 1 square Haat ≈ 2.25 sq ft
            return length * width * 2.25
        

        # CASE 2: Single numeric values followed by a unit (e.g., "14 aana", "4.5 aana", "2500 sq. feet")
        # Find the first floating point/integer number in the text
        frac_match = re.search(r'(\d+)\s*/\s*(\d+)', area_str)

        if frac_match:
            val = (
                float(frac_match.group(1))
                / float(frac_match.group(2))
            )   # for values like 1/2 Kattha
        else:
            num_match = re.search(r'[\d\.]+', area_str)

            if num_match:
                val = float(num_match.group())
            else:
                val = None

        if val is not None:
            
            if 'ropani' in area_str:
                return val * 5476
            elif 'aana' in area_str or 'ana' in area_str:
                return val * 342.25
            elif 'dhur' in area_str:
                return val * 182.25
            elif 'kattha' in area_str:
                return val * 3645
            elif 'bigha' in area_str:
                return val * 72900
            elif 'haat' in area_str:
                return val * 2.25
            elif 'sq. meter' in area_str or 'meter' in area_str:
                return val * 10.76
            elif 'sq. feet' in area_str or 'feet' in area_str:
                return val # Already in sq ft
            
        
        
        
    except Exception as e:
        # Fallback tracking if a row fails due to messy structural syntax
        return None
    
    return None

df['Road_Width_Feet'] = (
    df['Road Width']
    .astype(str)
    .str.extract(r'(\d+(?:\.\d+)?)')[0]
    .astype(float)
)

df['Road Type'] = df['Road Type'].astype(str).str.strip()
df['Road Type'] = df['Road Type'].replace({'nan': 'Unknown', 'N/A': 'Unknown'})
df['Road Type'] = df['Road Type'].fillna('Unknown')

df['Area_Corrected'] = df['Area'].apply(convert_area_to_sqft)

print(df['Road_Width_Feet'].head())
