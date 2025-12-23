
def process_RFM_per_tp(time_step, cut_off_date, promo_dataset_name, custom_folder_path, decay_coefficient):

    import pandas as pd
    import numpy as np
    import pickle
    import concurrent.futures
    import queue
    import os
    import warnings
    import json
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

    # Loading the processed RFM dataframes
    current_dir = os.path.dirname(__file__)  # Get the directory of the current script
    rfm_dfs_path = os.path.join(custom_folder_path, "rfm_dfs.pkl")
    with open(rfm_dfs_path, "rb") as f:
        rfm_dfs = pickle.load(f)

    # Get the column names
    config_file_path = os.path.join(custom_folder_path, "config.json")
    with open(config_file_path, "r") as f:
        config = json.load(f)
    id_col = config["id_col"]
    promo_date_col = config["promo_date_col"]


    # Preprocessing the Promotional Dataset:


    # Load the Promotional Dataset
    # Construct the file path dynamically to load the dataset from the "datasets" folder
    dataset_path = os.path.join(current_dir, "Datasets", promo_dataset_name)

    # Check the file extension and load the file accordingly
    if promo_dataset_name.endswith(".csv"):
        promoDataset = pd.read_csv(dataset_path, parse_dates=[promo_date_col], encoding='latin1')  # For CSV files
    elif promo_dataset_name.endswith(".xlsx"):
        promoDataset = pd.read_excel(dataset_path, parse_dates=[promo_date_col], engine='openpyxl')  # For Excel files
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")
    print("Promotional dataset loaded successfully!\n")

    # Remove duplicate rows from the DataFrame and reset the index
    promoDataset = promoDataset.drop_duplicates().reset_index(drop=True)

    print("Duplicate rows were removed from the Promotional dataset.\n")

    # Convert Dates to the Chosen Time-Step type
    promoDataset["TimePeriod"] = promoDataset[promo_date_col].dt.to_period(time_step)  # Convert to months, quarters, or years


    #promoDataset = promoDataset[promoDataset[promo_date_col] < str(cut_off_date)] # Filter out transactions from start of cut-off date onward
    promoDataset = promoDataset[[id_col, "TimePeriod"]]

    # Group to count how many promotions each (donor, TimePeriod) received.
    promo_counts = (
        promoDataset
        .groupby([id_col, "TimePeriod"])
        .size()
        .reset_index(name="promo_count")   # rename the aggregated column
    )


    tps = list(rfm_dfs.keys()) # TimePeriods
    # Convert each element in tps to a pd.Period
    tps = [pd.Period(tp, freq=time_step) for tp in tps]
    tp_min = min(tps)
    tp_max = max(tps)
    print(f"Considered Time Periods: {tp_min} - {tp_max}")
    print("*** The upper bound of the considered Time Periods may extend one period over the test dataset's cut-off date due to the model's logic. ***\n")


    
    # Getting the unique donors for each time period
    unique_donors = {}
    unique_donors_num = 0
    for i, tp in enumerate(tps):

        if i > 0:
                
            # Get all donors from current period
            current_donors = set(rfm_dfs[str(tp)][id_col].unique())
            
            # Collect all previously seen donors
            previous_donors = set()
            for j in range(i):
                previous_tp = tps[j]
                previous_donors.update(unique_donors[str(previous_tp)])
            
            # Find truly new donors in this period
            unique_donors[str(tp)] = np.array(list(current_donors - previous_donors))
        else:
            unique_donors[str(tp)] = rfm_dfs[str(tp)][id_col].unique()

        print(f"Found {len(unique_donors[str(tp)])} unique donors for time period {tp}.")
        unique_donors_num += len(unique_donors[str(tp)])
    
    print(f"\nFound {unique_donors_num} unique donors in total in the dataset's timeline.\n")




    # Reshaping the RFM tables in way that allows for dynamic RFM calculations
    # This entails forming a separate DataFrame for each donor
    donors_dyn = {}

    for i, tp in enumerate(tps):

        for donor in unique_donors[str(tp)]:
            
            # Initialize the Dynamic RFM DataFrame for each donor
            donors_dyn[donor] = pd.DataFrame(columns=["TimePeriod", "Recency", "Frequency", "MonetaryAvg", "u"])
            
            # Establish the TimePeriods column for each donor based on when they first donated
            donors_dyn[donor]["TimePeriod"] = pd.period_range(start=tp, end=tp_max, freq=time_step)


    donors_list = list(donors_dyn.keys())

    promo_counts = promo_counts[promo_counts[id_col].isin(donors_dyn.keys())]

    # Create a dictionary for quick and efficient lookups
    # key = (donor, timePeriod), value = promo_count
    promo_dict = {}
    for idx, row in promo_counts.iterrows():
        key = (row[id_col], row["TimePeriod"])
        promo_dict[key] = row["promo_count"]


    # Define the function that will process each donor
    # This function will be executed in parallel for each donor for efficiency
    print("\nProcessing the dynamic RFM DataFrames... This may take a while depending on the sizes of the datasets and the time-steps.\n")
    def process_donor(donor, donors_dyn, rfm_dfs, progress_queue, promo_dict, decay_coefficient):

        # Encoding dictionaries for the months
        month_encoding_1 = {1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 2.0, 6: 1.0, 7: 0.0, 8: -1.0, 9: -2.0, 10: -3.0, 11: -2.0, 12: -1.0}
        month_encoding_2 = {1: 3.0, 2: 2.0, 3: 1.0, 4: 0.0, 5: -1.0, 6: -2.0, 7: -3.0, 8: -2.0, 9: -1.0, 10: 0.0, 11: 1.0, 12: 2.0}


        # Iterate through each time period for this donor
        
        prev_v = 0 # We'll keep a running total of v

        for tp in donors_dyn[donor]["TimePeriod"]:
            # Extract the three RFM columns from rfm_dfs at TimePeriod tp
            donors_dyn[donor].loc[donors_dyn[donor]["TimePeriod"] == tp, ["Recency", "Frequency", "MonetaryAvg"]] = (
                rfm_dfs[str(tp)].loc[rfm_dfs[str(tp)][id_col] == donor, ["Recency", "Frequency", "MonetaryAvg"]].values
            )
            

            # Populate columns "u" and "v"
            mask = donors_dyn[donor]["TimePeriod"] == tp  
            
            # Lookup how many promos (instead of scanning promo_counts each time)
            match_count = promo_dict.get((donor, tp), 0) 
            
            # Update "u"
            donors_dyn[donor].loc[mask, "u"] = match_count
            
            # v = previous v + current u
            new_v = decay_coefficient * prev_v + match_count
            donors_dyn[donor].loc[mask, "v"] = new_v
            
            prev_v = new_v


        # Encoding the TimePeriod as time coordinates for later use
        # Add encoded_time_1 and encoded_time_2 columns
        donors_dyn[donor]["encoded_time_1"] = donors_dyn[donor]["TimePeriod"].dt.month.map(month_encoding_1)
        donors_dyn[donor]["encoded_time_2"] = donors_dyn[donor]["TimePeriod"].dt.month.map(month_encoding_2)

        # Reorder columns to place encoded_time_1 and encoded_time_2 after TimePeriod
        columns = (
            ["TimePeriod", "encoded_time_1", "encoded_time_2"]
            + [col for col in donors_dyn[donor].columns if col not in ["TimePeriod", "encoded_time_1", "encoded_time_2"]]
        )
        donors_dyn[donor] = donors_dyn[donor][columns]
        
        # Once processing is done for this donor, add a message to the progress queue
        progress_queue.put(donor)


    # Create a thread-safe queue for progress tracking
    progress_queue = queue.Queue()

    # Start the ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Use a list comprehension to submit tasks for all donors
        print(f"Using {executor._max_workers} threads for execution.")
        futures = [
            executor.submit(process_donor, donor, donors_dyn, rfm_dfs, progress_queue, promo_dict, decay_coefficient)
            for donor in donors_dyn.keys()
        ]
        
        # Track progress by getting items from the queue
        total_donors = len(donors_dyn)
        donors_processed = 0
        while donors_processed < total_donors:
            # This will block until an item is available in the queue
            progress_queue.get()  
            donors_processed += 1
            if donors_processed % 1000 == 0:
                print(f"Processed {donors_processed} out of {total_donors} donors...")







    # Populating the "y" and "y_lagged" columns in the dynamic RFM DataFrames

    # Loading the tp_y_dfs dataframes
    tp_y_dfs_path = os.path.join(custom_folder_path, "tp_y_dfs.pkl")
    with open(tp_y_dfs_path, "rb") as f:
        tp_y_dfs = pickle.load(f)

    last_tp = max(donors_dyn[donors_list[0]]["TimePeriod"])  # Get the last time period in the dataset

    i = 1
    for donor in donors_list:
        i +=1
        for tp in donors_dyn[donor]["TimePeriod"]:
            
            mask = donors_dyn[donor]["TimePeriod"] == tp # Boolean mask

            # Check if the donor is in the corresponding tp_y_dfs data
            if tp >= last_tp:
                donors_dyn[donor].loc[mask, "y"] = np.nan # Set the y value of the latest time period to NaN because we don't know the outcome yet
            elif donor in tp_y_dfs[str(tp)][id_col].values:
                donors_dyn[donor].loc[mask, "y"] = 1
            else:
                donors_dyn[donor].loc[mask, "y"] = 0
            

        # Reorder columns to make 'y' the last column
        columns = [col for col in donors_dyn[donor].columns if col != "y"] + ["y"]
        donors_dyn[donor] = donors_dyn[donor][columns]

    
        
    # Get the path to the "Deployment Files" folder inside the custom_folder_path  and save the deployment datasets
    deployment_folder = os.path.join(custom_folder_path, "Deployment Files")
    donors_dyn_deploy_path = os.path.join(deployment_folder, "donors_dyn.pkl")
    with open(donors_dyn_deploy_path, "wb") as f:
        pickle.dump(donors_dyn, f)
    

    # Convert unique_donors[str(last_tp)] to a set for efficient membership checking
    unique_donors_last_tp = set(unique_donors[str(last_tp)])
    # Removing the donors unique to the last time period for the train-test datasets
    donors_dyn = {donor: data for donor, data in donors_dyn.items() if donor not in unique_donors_last_tp}

    # Remove the last row of all values associated with every key in donors_dyn
    for donor, data in donors_dyn.items():
        donors_dyn[donor] = data.drop(data.index[-1])  # Drop the last row by its index


    # Save the final processed dynamic RFM DataFrames
    donors_dyn_path = os.path.join(custom_folder_path, "donors_dyn.pkl")
    with open(donors_dyn_path, "wb") as f:
        pickle.dump(donors_dyn, f)
  


    print(f"\nDonor-specific Train, Test, and Deployment RFM DataFrames based on the {'monthly' if time_step == 'M' else 'quarterly' if time_step == 'Q' else 'yearly'} time-steps have been saved successfully!\n")


def train_test_RFM_split(custom_folder_path, test_size):

    import pandas as pd
    import numpy as np
    import pickle
    import os
    from sklearn.model_selection import train_test_split

    # Construct the file path dynamically to load the dataset from the "datasets" folder
    # print(f"custom_folder_path: {custom_folder_path}")
    # Loading the processed Donor RFM dynamics dataframes
    donors_dyn_path = os.path.join(custom_folder_path, "donors_dyn.pkl")
    with open(donors_dyn_path, "rb") as f:
        donors_dyn = pickle.load(f)


    donors = list(donors_dyn.keys()) # list of donor keys

    # Split the keys into train and test sets
    train_keys, test_keys = train_test_split(donors, test_size=test_size) # The splitting is done based on the number of donors NOT their respective table sizes
    
    # Create two versions of donors_dyn
    donors_dyn_train = {key: donors_dyn[key] for key in train_keys}
    donors_dyn_test = {key: donors_dyn[key] for key in test_keys}

    # Save the train and test versions of donors_dyn
    donors_dyn_train_path = os.path.join(custom_folder_path, "donors_dyn_train.pkl")
    donors_dyn_test_path = os.path.join(custom_folder_path, "donors_dyn_test.pkl")

    with open(donors_dyn_train_path, "wb") as f:
        pickle.dump(donors_dyn_train, f)

    with open(donors_dyn_test_path, "wb") as f:
        pickle.dump(donors_dyn_test, f)


    print(f"Train and test datasets have been saved successfully!\n")