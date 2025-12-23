
def scale_dataset(custom_folder_path):

    import pandas as pd
    import numpy as np
    import pickle
    from sklearn.preprocessing import RobustScaler, MinMaxScaler
    import os

    # Construct the file path dynamically to load the dataset from the "datasets" folder

    # Loading the processed Donor RFM dynamics dataframes
    donors_dyn_test_path = os.path.join(custom_folder_path, "donors_dyn_test.pkl")
    with open(donors_dyn_test_path, "rb") as f:
        donors_dyn_test = pickle.load(f)

    donors_dyn_train_path = os.path.join(custom_folder_path, "donors_dyn_train.pkl")
    with open(donors_dyn_train_path, "rb") as f:
        donors_dyn_train = pickle.load(f)

    donors_train = list(donors_dyn_train.keys()) # list of train donor keys
    donors_test = list(donors_dyn_test.keys()) # list of test donor keys


    # Setting up empty arrays for scaling
    all_recency = np.empty((0, 1))
    all_frequency = np.empty((0, 1))
    all_money_ave = np.empty((0, 1))
    all_u = np.empty((0, 1))
    all_et1 = np.empty((0, 1))
    all_et2 = np.empty((0, 1))
    all_v = np.empty((0, 1))

    # Accumulating the values of the columns for all donors for scaling
    all_recency = np.concatenate([donors_dyn_train[donor]["Recency"].values for donor in donors_train])
    all_frequency = np.concatenate([donors_dyn_train[donor]["Frequency"].values for donor in donors_train])
    all_money_ave = np.concatenate([donors_dyn_train[donor]["MonetaryAvg"].values for donor in donors_train])
    all_u = np.concatenate([donors_dyn_train[donor]["u"].values for donor in donors_train])
    all_et1 = np.concatenate([donors_dyn_train[donor]["encoded_time_1"].values for donor in donors_train])
    all_et2 = np.concatenate([donors_dyn_train[donor]["encoded_time_2"].values for donor in donors_train])
    all_v = np.concatenate([donors_dyn_train[donor]["v"].values for donor in donors_train])

    # Initialize the scalers
    recency_scaler = MinMaxScaler(feature_range=(0, 1))
    frequency_scaler = MinMaxScaler(feature_range=(0, 1))
    money_ave_scaler = RobustScaler()
    u_scaler = MinMaxScaler(feature_range=(0, 1)) # Used minmax for u and v to scale because the robust scaler does not work well with the values of u and v
    et1_scaler = MinMaxScaler(feature_range=(-3, 3))
    et2_scaler = MinMaxScaler(feature_range=(-3, 3))
    v_scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit the scalers to the train dataset
    all_recency_scaled = recency_scaler.fit(all_recency.reshape(-1, 1))
    all_frequency_scaled = frequency_scaler.fit(all_frequency.reshape(-1, 1))
    all_money_ave_scaled = money_ave_scaler.fit(all_money_ave.reshape(-1, 1))
    all_u_scaled = u_scaler.fit(all_u.reshape(-1, 1))
    all_et1_scaled = et1_scaler.fit(all_et1.reshape(-1, 1))
    all_et2_scaled = et2_scaler.fit(all_et2.reshape(-1, 1))
    all_v_scaled = v_scaler.fit(all_v.reshape(-1, 1))

    donors_dyn_train_scaled = donors_dyn_train.copy() # Make a deep copy of the original dataset for scaling
    donors_dyn_test_scaled = donors_dyn_test.copy() # Make a deep copy of the original dataset for scaling

    # Scaling the columns for each donor for the train and test datasets
    for donor in donors_train:
        donors_dyn_train_scaled[donor]["Recency"] = recency_scaler.transform(donors_dyn_train_scaled[donor]["Recency"].values.reshape(-1, 1)) 
        donors_dyn_train_scaled[donor]["Frequency"] = frequency_scaler.transform(donors_dyn_train_scaled[donor]["Frequency"].values.reshape(-1, 1)) 
        donors_dyn_train_scaled[donor]["MonetaryAvg"] = money_ave_scaler.transform(donors_dyn_train_scaled[donor]["MonetaryAvg"].values.reshape(-1, 1)) 
        donors_dyn_train_scaled[donor]["u"] = u_scaler.transform(donors_dyn_train_scaled[donor]["u"].values.reshape(-1, 1))
        donors_dyn_train_scaled[donor]["encoded_time_1"] = et1_scaler.transform(donors_dyn_train_scaled[donor]["encoded_time_1"].values.reshape(-1, 1))
        donors_dyn_train_scaled[donor]["encoded_time_2"] = et2_scaler.transform(donors_dyn_train_scaled[donor]["encoded_time_2"].values.reshape(-1, 1)) 
        donors_dyn_train_scaled[donor]["v"] = v_scaler.transform(donors_dyn_train_scaled[donor]["v"].values.reshape(-1, 1))

    for donor in donors_test:
        donors_dyn_test_scaled[donor]["Recency"] = recency_scaler.transform(donors_dyn_test_scaled[donor]["Recency"].values.reshape(-1, 1)) 
        donors_dyn_test_scaled[donor]["Frequency"] = frequency_scaler.transform(donors_dyn_test_scaled[donor]["Frequency"].values.reshape(-1, 1)) 
        donors_dyn_test_scaled[donor]["MonetaryAvg"] = money_ave_scaler.transform(donors_dyn_test_scaled[donor]["MonetaryAvg"].values.reshape(-1, 1)) 
        donors_dyn_test_scaled[donor]["u"] = u_scaler.transform(donors_dyn_test_scaled[donor]["u"].values.reshape(-1, 1))
        donors_dyn_test_scaled[donor]["encoded_time_1"] = et1_scaler.transform(donors_dyn_test_scaled[donor]["encoded_time_1"].values.reshape(-1, 1))
        donors_dyn_test_scaled[donor]["encoded_time_2"] = et2_scaler.transform(donors_dyn_test_scaled[donor]["encoded_time_2"].values.reshape(-1, 1)) 
        donors_dyn_test_scaled[donor]["v"] = v_scaler.transform(donors_dyn_test_scaled[donor]["v"].values.reshape(-1, 1)) 

    # Save the donors_dyn_scaled DataFrames
    donors_dyn_train_scaled_path = os.path.join(custom_folder_path, "donors_dyn_train_scaled.pkl")
    with open(donors_dyn_train_scaled_path, "wb") as f:
        pickle.dump(donors_dyn_train_scaled, f)

    donors_dyn_test_scaled_path = os.path.join(custom_folder_path, "donors_dyn_test_scaled.pkl")
    with open(donors_dyn_test_scaled_path, "wb") as f:
        pickle.dump(donors_dyn_test_scaled, f)

    # Save the scalers in a dictionary
    scalers = {
        "recency_scaler": recency_scaler,
        "frequency_scaler": frequency_scaler,
        "money_ave_scaler": money_ave_scaler,
        "u_scaler": u_scaler,
        "et1_scaler": et1_scaler,
        "et2_scaler": et2_scaler,
        "v_scaler": v_scaler
        }
    
    scalers_path = os.path.join(custom_folder_path, "dataset_scalers.pkl")
    with open(scalers_path, "wb") as f:
        pickle.dump(scalers, f)


    print(f"Scaled train and test RFM tables and the dataset scalers have been saved successfully!\n")
