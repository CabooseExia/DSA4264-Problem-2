from pathlib import Path
import os
import pandas as pd

app_dir = Path(__file__).parent

filepath = f'{app_dir}\\wordclouds\\r_Singapore.png'

# wordclouds/r_Singpapore.png
if os.path.exists(filepath):
    print('exists')

print(app_dir)