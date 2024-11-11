import pandas as pd

df = pd.read_csv('../data/200_expertly_chosen_texts.csv')

df['toxic'] = False
df['hateful'] = False

current_row = 0

# Instructions for the user
print("Labeling options:")
print("1: Neither toxic nor hateful")
print("2: Toxic only")
print("3: Hateful only")
print("4: Both toxic and hateful")
print("Press 'q' at any time to quit.\n")

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    print(f"Text: {row['text']}")
    
    # Get user input for labeling
    user_input = input("Enter your label (1/2/3/4): ").lower()
    
    # Exit if the user enters 'q'
    if user_input == 'q':
        print("Exiting early...")
        break
    
    # Apply labeling based on the user's input
    if user_input == '1':
        df.at[index, 'toxic'] = False
        df.at[index, 'hateful'] = False
    elif user_input == '2':
        df.at[index, 'toxic'] = True
        df.at[index, 'hateful'] = False
    elif user_input == '3':
        df.at[index, 'toxic'] = False
        df.at[index, 'hateful'] = True
    elif user_input == '4':
        df.at[index, 'toxic'] = True
        df.at[index, 'hateful'] = True
    else:
        print("Invalid input. Please enter 1, 2, 3, 4, or 'q' to quit.")
    


# After labeling, save the DataFrame to a CSV file
df.to_csv('data/labeled_data.csv', index=False)
print("Labeled data saved to 'labeled_data.csv'.")