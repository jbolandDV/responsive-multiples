
def window_construction(custom_folder_path, model_window_size):

    # Input: custom_folder_path, model_window_size

    import pandas as pd
    import numpy as np
    import pickle
    import os
    from sklearn.utils import shuffle

    print(f"Constructing windows for training and testing...\n")

    # Loading the split and scaled Donor RFM dynamics dataframes
    donors_dyn_train_scaled_path = os.path.join(custom_folder_path, "donors_dyn_train_scaled.pkl")
    with open(donors_dyn_train_scaled_path, "rb") as f:
        donors_dyn_train = pickle.load(f)
        
    donors_dyn_test_scaled_path = os.path.join(custom_folder_path, "donors_dyn_test_scaled.pkl")
    with open(donors_dyn_test_scaled_path, "rb") as f:
        donors_dyn_test = pickle.load(f)


    scalers_path = os.path.join(custom_folder_path, "dataset_scalers.pkl")
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    et1_scaler = scalers["et1_scaler"]
    et2_scaler = scalers["et2_scaler"]


    def construct_windows(donors_dyn, model_window_size):
        transformed_donors = {}

        for donor_id, donor_data in donors_dyn.items():
            donor_windows = []  # To store 12-row windows for this donor

            # Add the is_padded column to the donor's DataFrame
            donor_data["is_padded"] = 0

            # Reorder columns to make is_padded the second-to-last column
            columns = (
                [col for col in donor_data.columns if col != "y"] + ["y"]
            )
            donor_data = donor_data[columns]

            """
            START: Pre-Padding
            ------------------------------------------------------------
            # The following code is used to ensure that the first row of the first window starts with January.
            # It pads the beginning of the DataFrame with zeros for the months before January.
            # Use with discretion and only use if pre-padding is your intention.
            ------------------------------------------------------------
            """

            # # Ensure the first row of the first window starts with January
            # first_month = donor_data["TimePeriod"].iloc[0].month
            
            # if first_month != 1:
            #     # Pad rows in the beginning to make the first month January
            #     # Calculate the number of months to pad
            #     months_to_pad = first_month - 1

            #     # Create a padding DataFrame with the same columns as donor_data
            #     padding = pd.DataFrame(
            #         {col: 0 for col in donor_data.columns},  # Fill with zeros
            #         index=range(months_to_pad)
            #     )
            #     padding["is_padded"] = 1  # Mark these rows as padded
            #     donor_data = pd.concat([padding, donor_data], ignore_index=True)


            """
            ------------------------------------------------------------
            ------------------------------------------------------------
            END: Pre-Padding
            """
                

            # Split the donor's DataFrame into 12-row windows
            for start_idx in range(0, len(donor_data), model_window_size):
                window = donor_data.iloc[start_idx:start_idx + model_window_size].copy()

                # Reset the index for the window
                window.reset_index(drop=True, inplace=True)

                # Check if the window has fewer than 12 rows
                if len(window) < model_window_size:
                    # Calculate the number of rows to pad
                    rows_to_pad = model_window_size - len(window)

                    # Create a padding DataFrame
                    padding = pd.DataFrame(
                        {col: 0 for col in donor_data.columns},  # Fill with zeros
                        index=range(rows_to_pad)
                    )
                    padding["is_padded"] = 1  # Mark padded rows

                    # Append the padding to the window
                    window = pd.concat([window, padding], ignore_index=True)

                # Time encoding dictionaries for the months (based on indexes NOT the month numbers) + Scaling them because the padded encoded_time_1 and encoded_time_2 columns are not scaled yet
                month_encoding_1_idx = {0: et1_scaler.transform([[0]]), 
                                        1: et1_scaler.transform([[1]]), 
                                        2: et1_scaler.transform([[2]]), 
                                        3: et1_scaler.transform([[3]]), 
                                        4: et1_scaler.transform([[2]]), 
                                        5: et1_scaler.transform([[1]]), 
                                        6: et1_scaler.transform([[0]]), 
                                        7: et1_scaler.transform([[-1]]), 
                                        8: et1_scaler.transform([[-2]]), 
                                        9: et1_scaler.transform([[-3]]), 
                                        10: et1_scaler.transform([[-2]]), 
                                        11: et1_scaler.transform([[-1]])}
                
                month_encoding_2_idx = {0: et2_scaler.transform([[3]]), 
                                        1: et2_scaler.transform([[2]]), 
                                        2: et2_scaler.transform([[1]]), 
                                        3: et2_scaler.transform([[0]]), 
                                        4: et2_scaler.transform([[-1]]), 
                                        5: et2_scaler.transform([[-2]]), 
                                        6: et2_scaler.transform([[-3]]), 
                                        7: et2_scaler.transform([[-2]]), 
                                        8: et2_scaler.transform([[-1]]), 
                                        9: et2_scaler.transform([[0]]), 
                                        10: et2_scaler.transform([[1]]), 
                                        11: et2_scaler.transform([[2]])}

                padded_rows_idx = []  # To store the indexes of padded rows for each window
                for idx in range(len(window)):

                    if window["is_padded"].iloc[idx] == 1:
                        # Add encoded_time_1 and encoded_time_2 columns

                        window.loc[idx, "TimePeriod"] = window.loc[idx-1, "TimePeriod"] + 1 # Current TimePeriod is the previous TimePeriod + 1 month because in the padded rows TimePeriods are not added
                        
                        window.loc[idx, "encoded_time_1"] = month_encoding_1_idx[window.loc[idx, "TimePeriod"].month - 1]
                        window.loc[idx, "encoded_time_2"] = month_encoding_2_idx[window.loc[idx, "TimePeriod"].month - 1]

                        
                        padded_rows_idx.append(idx)
                
                """
                ------------------------------------------------------------
                ------------------------------------------------------------
                """
                # Making sure that if a table is pre-padded, the y label for the last padded row is 1
                # This only necessary when the windows are pre-padded so row indexes correspond to month numbers
                # Check if the list is empty
                if not padded_rows_idx:
                    last_index_before_jump = None  # Or handle the empty case as needed
                elif padded_rows_idx[0] == 0:                    
                    # Initialize the last index before the jump
                    last_index_before_jump = padded_rows_idx[-1]  # Default to the last element
                    # Find the last index before the jump
                    for i in range(1, len(padded_rows_idx)):
                        if padded_rows_idx[i] - padded_rows_idx[i - 1] > 1:
                            last_index_before_jump = padded_rows_idx[i - 1]
                            break

                    window.loc[last_index_before_jump, "y"] = 1
                """
                ------------------------------------------------------------
                ------------------------------------------------------------
                """
                                  
                    

                # Append the 12-row window to the donor's list of windows
                donor_windows.append(window)

            # Store the transformed windows for this donor
            transformed_donors[donor_id] = donor_windows

        return transformed_donors


    # Apply the transformation to train and test datasets
    train_windows = construct_windows(donors_dyn_train, model_window_size)
    test_windows = construct_windows(donors_dyn_test, model_window_size)

    # Save the transformed windows for later use
    train_windows_path = os.path.join(custom_folder_path, "train_windows.pkl")
    test_windows_path = os.path.join(custom_folder_path, "test_windows.pkl")

    with open(train_windows_path, "wb") as f:
        pickle.dump(train_windows, f)

    with open(test_windows_path, "wb") as f:
        pickle.dump(test_windows, f)

    print(f"Train and test windows have been saved successfully!\n")


    def prepare_features_labels(windows):
        X = []  # List of feature DataFrames (each representing a window)
        y = []  # List of label arrays (each representing a label set)

        for donor_id, donor_windows in windows.items():
            for window in donor_windows:
                # Separate features (X) and labels (y)
                X.append(window.drop(columns=["y", "TimePeriod", "u"]))  # Keep as DataFrame
                y.append(window["y"])  # Keep as Series
        
        # Shuffle X and y together while maintaining correspondence
        X, y = shuffle(X, y)

        return X, y


    # Prepare train and test datasets
    X_train, y_train = prepare_features_labels(train_windows)
    X_test, y_test = prepare_features_labels(test_windows)

    print(f"Number of train feature windows: {len(X_train)}, Number of train label windows: {len(y_train)}\n")
    print(f"Number of test feature windows: {len(X_test)}, Number of test label windows: {len(y_test)}\n")

    # Save the prepared features and labels
    X_train_path = os.path.join(custom_folder_path, "X_train.pkl")
    y_train_path = os.path.join(custom_folder_path, "y_train.pkl")
    X_test_path = os.path.join(custom_folder_path, "X_test.pkl")
    y_test_path = os.path.join(custom_folder_path, "y_test.pkl")

    with open(X_train_path, "wb") as f:
        pickle.dump(X_train, f)

    with open(y_train_path, "wb") as f:
        pickle.dump(y_train, f)

    with open(X_test_path, "wb") as f:
        pickle.dump(X_test, f)

    with open(y_test_path, "wb") as f:
        pickle.dump(y_test, f)

    print(f"Train and test feature windows and label windows have been saved successfully!\n")


def window_construction_deploy(custom_folder_path, model_window_size):
    # Input: custom_folder_path, model_window_size

    import pandas as pd
    import numpy as np
    import pickle
    import os
    from concurrent.futures import ProcessPoolExecutor
    import concurrent.futures
    import queue
    from multiprocessing import Manager
    import joblib

    print(f"Constructing windows for deployment...\n")

    # Loading the split and scaled Donor RFM dynamics dataframes
    donors_dyn_path = os.path.join(custom_folder_path, "Deployment Files", "donors_dyn.pkl")
    with open(donors_dyn_path, "rb") as f:
        donors_dyn = pickle.load(f)

    scalers_path = os.path.join(custom_folder_path, "dataset_scalers.pkl")
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    et1_scaler = scalers["et1_scaler"]
    et2_scaler = scalers["et2_scaler"]    


    def construct_windows_deploy(donors_dyn, model_window_size):
        donor_windows_perms = {key: None for key in donors_dyn.keys()} # Initialize a dictionary to store the different permutations of windows for each donor

        print(f"Number of donors: {len(list(donors_dyn.keys()))}\n")
        tot_num_windows = 0 # To calculate the total number of windows for deployment
        i = 0 # Counter for the number of donors processed
        for donor_id, donor_data in donors_dyn.items():
            i = i + 1 # Increment the counter for the number of donors processed
            (print(f"Processing donor {i} of {len(list(donors_dyn.keys()))}...\n") if i % 1000 == 0 else None) # Print the number of donors processed every 1000 donors
            donor_windows_perms[donor_id] = {key: None for key in list( range( 1, min(len(donor_data)+1, (model_window_size)+1 ) ) ) } # Store the different permutations of windows for this donor
            """
            For donor_windows_perms the idea is to make a dictionary for each donor. Each donor's dictionary will have keys from 1 to the number of real rows it has.
            For example, if a donor has 5 months worth of data up to the deployment cut-off date, their dictionary will have five keys from 1 to 5.
            If they have got more than model_window_size months worth of data, their dictionary will have model_window_size - 1 keys from 1 to model_window_size - 1.
            The idea is for the ML model to make predictions for based on the model_window_size and how many months worth of data the donor has got.
            If they have 1 month of data, there is only one way for the model to provide donation probabilities, i.e. setting it as the first row and predicting the remaining model_window_size - 1 row labels.
            If they have let's say 5 months of data, there are five ways for doing so. Using only the latest row to predict the next model_window_size - 1 rows' labels. Using only the latest 2 rows to predict the next model_window_size - 2 rows' labels. And so on and so forth.
            donor_windows_perms saves the different permutations of windows for each donor for deployment later.

            """

            # Add the is_padded column to the donor's DataFrame
            donor_data["is_padded"] = 0

            # Reorder columns to make is_padded the second-to-last column
            columns = (
                [col for col in donor_data.columns if col != "y"] + ["y"]
            )
            donor_data = donor_data[columns]

            # Extract the donor's last perm_len rows to turn into {model_window_size} sized windows
            for perm_len in list(donor_windows_perms[donor_id].keys()):

                window = donor_data.iloc[-perm_len:].copy()

                # Reset the index for the window
                window.reset_index(drop=True, inplace=True)

                # Check if the window has fewer than 12 rows
                if len(window) < model_window_size:
                    # Calculate the number of rows to pad
                    rows_to_pad = model_window_size - len(window)

                    # Create a padding DataFrame
                    padding = pd.DataFrame(
                        {col: 0 for col in donor_data.columns},  # Fill with zeros
                        index=range(rows_to_pad)
                    )
                    padding["is_padded"] = 1  # Mark padded rows. Note: it's zero here because we are passing this into the model for predictions.

                    # Append the padding to the window
                    window = pd.concat([window, padding], ignore_index=True)

                # Time encoding dictionaries for the months (based on indexes NOT the month numbers) + Scaling them because the padded encoded_time_1 and encoded_time_2 columns are not scaled yet
                month_encoding_1_idx = {0: et1_scaler.transform([[0]]), 
                                        1: et1_scaler.transform([[1]]), 
                                        2: et1_scaler.transform([[2]]), 
                                        3: et1_scaler.transform([[3]]), 
                                        4: et1_scaler.transform([[2]]), 
                                        5: et1_scaler.transform([[1]]), 
                                        6: et1_scaler.transform([[0]]), 
                                        7: et1_scaler.transform([[-1]]), 
                                        8: et1_scaler.transform([[-2]]), 
                                        9: et1_scaler.transform([[-3]]), 
                                        10: et1_scaler.transform([[-2]]), 
                                        11: et1_scaler.transform([[-1]])}
                
                month_encoding_2_idx = {0: et2_scaler.transform([[3]]), 
                                        1: et2_scaler.transform([[2]]), 
                                        2: et2_scaler.transform([[1]]), 
                                        3: et2_scaler.transform([[0]]), 
                                        4: et2_scaler.transform([[-1]]), 
                                        5: et2_scaler.transform([[-2]]), 
                                        6: et2_scaler.transform([[-3]]), 
                                        7: et2_scaler.transform([[-2]]), 
                                        8: et2_scaler.transform([[-1]]), 
                                        9: et2_scaler.transform([[0]]), 
                                        10: et2_scaler.transform([[1]]), 
                                        11: et2_scaler.transform([[2]])}

                padded_rows_idx = []  # To store the indexes of padded rows for each window
                for idx in range(len(window)):

                    if window["is_padded"].iloc[idx] == 1:

                        # Add encoded_time_1 and encoded_time_2 columns

                        window.loc[idx, "TimePeriod"] = window.loc[idx-1, "TimePeriod"] + 1 # Current TimePeriod is the previous TimePeriod + 1 month because in the padded rows TimePeriods are not added
                        
                        window.loc[idx, "encoded_time_1"] = month_encoding_1_idx[window.loc[idx, "TimePeriod"].month - 1]
                        window.loc[idx, "encoded_time_2"] = month_encoding_2_idx[window.loc[idx, "TimePeriod"].month - 1]
                        window.loc[idx, "Recency"] = window.loc[idx-1, "Recency"] + 1
                        window.loc[idx, "Frequency"] = window.loc[idx-1, "Frequency"]  # We assume that in the future we are trying to make predictions for, frequency and monetary avg are not going to change because there are no donations
                        window.loc[idx, "MonetaryAvg"] = window.loc[idx-1, "MonetaryAvg"]
                        window.loc[idx, "y"] = np.nan  # Set y to NaN for future rows

                        padded_rows_idx.append(idx)
                
                                  
                    

                # Store the 12-row window as the donor's window with perm_len real rows
                # Separate features (X) and labels (y)
                X = window.drop(columns=["y"])  # Keep as DataFrame and DO NOT DROP u column
                y = window["y"]  # Keep as Series
                donor_windows_perms[donor_id][perm_len] = []
                donor_windows_perms[donor_id][perm_len].append(X)  # Append the feature DataFrame to the donor's list of feature DataFrames
                donor_windows_perms[donor_id][perm_len].append(y)  # Append the label Series to the donor's list of label Series

                tot_num_windows = tot_num_windows + 1 # Increment the total number of windows for deployment


        return donor_windows_perms, tot_num_windows

    
    # Apply the transformation to deployment datasets
    deploy_donor_windows, tot_num_windows = construct_windows_deploy(donors_dyn, model_window_size)


    deploy_files_path = os.path.join(custom_folder_path, "Deployment Files")
    deploy_windows_path = os.path.join(deploy_files_path, "deploy_donor_windows.joblib")

    print(f"\nSaving deployment windows to {deploy_windows_path}...\n")
    print("\nThis may take a while depending on the number of donors and the size of the windows.\n")
    joblib.dump(deploy_donor_windows, deploy_windows_path)

    print(f"Deployment windows have been saved successfully!\n")
    print(f"Total Number of deployment windows: {tot_num_windows}\n")

