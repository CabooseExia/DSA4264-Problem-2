import numpy as np
import ast
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
from nltk.corpus import stopwords

#reading the data
df = pd.DataFrame()

chunk_size = 10000
for chunk in pd.read_csv('../data/Reddit-Threads_2020-2021.csv', chunksize=chunk_size):
    print(chunk.head())  
    df = pd.concat([df, chunk])
for chunk in pd.read_csv('../data/Reddit-Threads_2022-2023.csv', chunksize=chunk_size):
    print(chunk.head())  
    df = pd.concat([df, chunk])



# Cleaning the data

# df['moderation'] = df['moderation'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
# moderation_dicts = df['moderation']
# moderation_normalized = pd.json_normalize(moderation_dicts)
# # print(moderation_normalized)
# df = df.reset_index(drop=True)
# moderation_normalized = moderation_normalized.reset_index(drop=True)
# df_normalized = pd.concat([df.drop(columns=['moderation']), moderation_normalized], axis=1)
# print(df_normalized.columns)
df_normalized = df

### removing deleted or removed text ###
df_normalized = df_normalized[df_normalized['text'] != '[deleted]']
df_normalized = df_normalized[df_normalized['text'] != '[removed]']
### removing deleted or removed text ###

### stop word removal ###
# stop_words = set(stopwords.words('english'))

# def remove_stop_words(text):
#     if isinstance(text, str):  # Check if the text is a string
#         return ' '.join([word for word in text.split() if word.lower() not in stop_words])
#     return text 

# df_normalized['text'] = df_normalized['text'].apply(remove_stop_words)
# print(df_normalized['text'])
# print(stop_words)
### stop word removal ###


# Sample the DataFrame to make it manageable
df_sampled = df_normalized.sample(n=10000, random_state=42)

# Initialize a list to store indices of selected rows
selected_rows = []
selected_count = 0

# Iterate through each row in the sampled DataFrame
for i, row in df_sampled.iterrows():
    print(f"Row {i}:")
    print(row['text'])  # Adjust this to display the relevant column from your DataFrame
    
    # Prompt the user to decide if they want to select this row
    user_input = input("Select this row? (y/n) or (q to quit): ").lower()
    
    # If the user chooses 'y', add the index to the selected rows list
    if user_input == 'y':
        selected_rows.append(i)
        selected_count += 1
        print(f"Rows selected so far: {selected_count}")
    
    # Option to quit the process early
    elif user_input == 'q':
        print("Exiting early...")
        break

# After finishing, create a DataFrame of the selected rows
selected_rows_df = df_sampled.loc[selected_rows]

# Save the selected rows DataFrame to a CSV file
selected_rows_df.to_csv('../data/expertly_selected_data.csv', index=False)

# Display the selected rows DataFrame and confirmation of saving
print(f"\nSelected {selected_count} rows in total.")
print("Data has been saved to 'selected_rows.csv'.")
selected_rows_df.head()