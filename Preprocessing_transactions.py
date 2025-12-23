
def process_transactions(time_step, filter_unique_ids, cut_off_date, transaction_dataset_name, custom_folder_path, exclude_one_time_donors):
    import pandas as pd
    import numpy as np
    import pickle
    import os
    import json

    """
    Processes the transaction dataset to calculate RFM metrics and prepare data for modeling.

    Args:
        time_step (str): Time step for aggregation ('M' for months, 'Q' for quarters, 'Y' for years).
        filter_unique_ids (float): Fraction of unique IDs to retain.
        cut_off_date (str): Date to filter out transactions from then onwards.
        transaction_dataset_name (str): Name of the transaction dataset file.

    Returns:
        None
    """
    # window_size = (12-1) if time_step == "M" else (4-1) if time_step == "Q" else (1-1) # Determining the window size based on the time-step
    window_size = (60-1) if time_step == "M" else (4-1) if time_step == "Q" else (1-1) # Determining the window size based on the time-step

    # Construct the file path dynamically to load the dataset from the "datasets" folder
    current_dir = os.path.dirname(__file__)  # Get the directory of the current script
    dataset_path = os.path.join(current_dir, "Datasets", transaction_dataset_name)  # Path to the dataset file


    # Get the pre-defined configuration variables from the config.json file
    config_file_path = os.path.join(custom_folder_path, "config.json")
    with open(config_file_path, "r") as f:
        config = json.load(f)
    id_col = config["id_col"]
    tran_date_col = config["tran_date_col"]
    amount_col = config["amount_col"]
    donation_type_col = config["donation_type_col"]
    pattern = config["pattern"]
    match_pattern = config["match_pattern"]
    exclude_nans = config["exclude_nans"]
 

    # Check the file extension and load the file accordingly
    if transaction_dataset_name.endswith(".csv"):
        tranDataset = pd.read_csv(dataset_path, parse_dates=[tran_date_col], encoding='latin1')  # For CSV files
    elif transaction_dataset_name.endswith(".xlsx"):
        tranDataset = pd.read_excel(dataset_path, parse_dates=[tran_date_col], engine='openpyxl')  # For Excel files
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")
    
    print("\nTransactions dataset loaded successfully!\n")
    # Convert Dates to the Chosen Time-Step
    tranDataset["TimePeriod"] = tranDataset[tran_date_col].dt.to_period(time_step)  # Convert to months, quarters, or years


    # Remove dollar signs and commas, then convert to float
    tranDataset[amount_col] = tranDataset[amount_col].replace('[\$,]', '', regex=True).astype(float)

    
    # Remove rows with zero amounts
    tranDataset = tranDataset[tranDataset[amount_col] != 0]
    print("\nFiltered out the 0-dollar transactions.\n")



    # Filter the subscribers out of the dataset and include only the irregular donors

    def get_exclude_condition(
        df, 
        donation_type_col, 
        pattern, 
        match_pattern=True, 
        exclude_nans=True, 
        case=False, 
        regex=True
    ):
        """
        Returns a boolean Series where True indicates rows to exclude based on:
        - pattern: regex or string pattern to match
        - match_pattern: if True, exclude rows matching the pattern; if False, exclude rows NOT matching the pattern
        - exclude_nans: if True, exclude NaN rows regardless of pattern; if False, do not exclude NaNs regardless of pattern
        - case: case sensitivity for pattern matching
        - regex: use regex for pattern matching
        """
        cond = df[donation_type_col].str.contains(pattern, case=case, na=False, regex=regex)
        cond = cond if match_pattern else ~cond

        if exclude_nans:
            cond = cond | df[donation_type_col].isna()
        else:
            cond = cond & ~df[donation_type_col].isna()

        return cond

    excluded_ids = tranDataset[get_exclude_condition(tranDataset, donation_type_col, pattern, match_pattern, exclude_nans)][id_col].unique()

    # # Identify donor IDs with "friends of the Food bank", "monthly", or NaN in the donation_type_col column
    # excluded_ids = tranDataset[
    #     tranDataset[donation_type_col].str.contains("friends of the Food bank|monthly|friend of the Food bank|August 2021 Sustainer Blitz Email Appeal", case=False, na=False) |
    #     tranDataset[donation_type_col].isna()
    # ][id_col].unique()

    print(
    f"\nExcluded {len(excluded_ids)} donor IDs "
    f"{'matching' if match_pattern else 'not matching'} the pattern \"{pattern}\", and "
    f"{'including' if not exclude_nans else 'excluding'} the NaN rows in the '{donation_type_col}' column.\n"
    )

    # Remove all rows associated with those donor IDs
    filtered_tranDataset = tranDataset[~tranDataset[id_col].isin(excluded_ids)]
    tranDataset = filtered_tranDataset


    unique_ids_prev = tranDataset[id_col].unique()

    if exclude_one_time_donors == 1:
        # Group by id_col and filter out IDs with only one occurrence (Those who donated only once)
        tranDataset = tranDataset.groupby(id_col).filter(lambda x: len(x) > 1)

    one_off_donors = tranDataset.groupby(id_col).filter(lambda x: len(x) == 1)  # Get one-off donors (those who donated only once)
    one_off_ids = one_off_donors[id_col].unique().tolist()

    # Add a one_off_ids variable to the config dictionary
    config["one_off_ids"] = one_off_ids  

    # Save the updated config back to the file
    with open(config_file_path, "w") as f:
        json.dump(config, f, indent=4)

    # Get a list of unique IDs after removing the one-off donors
    unique_ids = tranDataset[id_col].unique()

    print(f"\nExcluded {len(unique_ids_prev)-len(unique_ids)} donor IDs that only had one donation.\n")



    # Randomly select a portion of the unique IDs
    sampled_ids = pd.Series(unique_ids).sample(frac=filter_unique_ids) 

    # Filter the dataset to include only rows with the sampled IDs
    filtered_tranDataset = tranDataset[tranDataset[id_col].isin(sampled_ids)]

    tranDataset = filtered_tranDataset

    # Print the result
    print(f"Reduced the transactions dataset to include only {filter_unique_ids*100}% of the unique IDs.\n")



    tranDataset = tranDataset.sort_values(["TimePeriod"])  # Sort by donor & date
    tranDataset_y = tranDataset.copy(deep=True)  # Keep a copy of the original dataset for later use

    tranDataset = tranDataset[tranDataset[tran_date_col] < str(cut_off_date)] # Filter out transactions from start of cut-off date month onwards
    # tranDataset_y = tranDataset_y[tranDataset_y[tran_date_col] < "2025-07-01"] # This is used for calculating the binary variable "y". It needs to be one time-step ahead

    # Get unique TimePeriods in ascending order
    unique_tp = tranDataset["TimePeriod"].unique()
    unique_tp_y = tranDataset_y["TimePeriod"].unique()


    # Create a dictionary to store DataFrames for each Time-Step (ts)
    tp_dfs = {}

    # Iterate over each unique TimePeriod and create DataFrames
    for i, ts in enumerate(unique_tp):
        # Select data up to the current TimePeriod
        tp_data = tranDataset[tranDataset["TimePeriod"] <= ts]
        # Store the DataFrame in the dictionary
        tp_dfs[str(ts)] = tp_data


    # Create a dictionary to store DataFrames for each Time-Step (ts) (For later use in the prediction model for getting the binary donation status)
    tp_y_dfs = {}

    # Iterate over each unique TimePeriod and create DataFrames
    for i, ts in enumerate(unique_tp_y):
        # Select data up to the current TimePeriod
        tp_y_data = tranDataset_y[tranDataset_y["TimePeriod"] == ts]
        # Drop the irrelevant columns
        tp_y_data = tp_y_data[[id_col, "TimePeriod",]]
        # Store the DataFrame in the dictionary
        tp_y_dfs[str(ts)] = tp_y_data

    # Save the tp_y_dfs to the "files" folder
    tp_y_dfs_path = os.path.join(custom_folder_path, "tp_y_dfs.pkl")
    with open(tp_y_dfs_path, "wb") as f:   # Save the dataset for later use
        pickle.dump(tp_y_dfs, f)

    print(f"IDs of the donors that paid in each {'month' if time_step == 'M' else 'quarter' if time_step == 'Q' else 'year'} have been saved successfully!\n")






    def calculate_RFM(tp_df):

        """
        This function calculates and updates the Recency, Frequency, and Monetary Value at each TimePeriod for all donors.
        If a certain donor has not made a transaction yet, their RFM will not be calculated until they do. After they have donated for the first time,
        their RFM values will be updated for each TimePeriod and while only looking back at the prior year.

        Recency looks at the entire timeline instead of only the past year.
        """
        # Aggregate Data Per Donor & Last donation date
        rfm_r = tp_df.groupby([id_col]).agg(
            LastDonation=(tran_date_col, "max")  # Most recent donation
        ).reset_index()

        ## Step 1: Compute Recency (Time Since Last Donation):

        # Determine the end of the TimePeriod (e.g. end of month or quarter or year) for the maximum TimePeriod in tp_df
        end_of_period = tp_df["TimePeriod"].max().to_timestamp(how='end')

        # Calculate the difference between the end-of-TimePeriod date and the LastDonation date
        rfm_r["Recency"] = (end_of_period - rfm_r["LastDonation"]).dt.days // (30 if time_step == "M" else 90 if time_step == "Q" else 365)
        rfm_r["Recency"] = rfm_r["Recency"].fillna(0).astype(int)  # Ensure first donation gets recency of 0
        # Drop the LastDonation column from rfm_r
        rfm_r = rfm_r.drop(columns=["LastDonation"])


        # Step 2: Compute Frequency and Monetary Value:

        # Compute Rolling Monetary Value (M_it as the Last Number of Periods' Average) 
        # and Frequency (for the last 12 months, or 4 quarters, or 1 year - decided based on the window_size variable)

        rfm_fm = tp_df.groupby([id_col, "TimePeriod"]).agg(
            Frequency=(amount_col, "count"),  # Count donations in the period
            Monetary=(amount_col, "sum")  # Total donation amount
        ).reset_index()
        rfm_fm = rfm_fm.sort_values([id_col, "TimePeriod"])  # Sort by donor & time before applying rolling mean. *** This is important because the rolling mean is calculated based on the sorted order of the data. ***

        
        timeperiod_max = tp_df["TimePeriod"].max()

        def compute_fm(row):
            # This accept one row and compute frequency and monetary values for that row
            current_time = row["TimePeriod"]

            # Calculate the difference in months/quarters/years between the current donation and the latest time period
            time_diff = (timeperiod_max - current_time).n  # This gives the difference in time-steps

            # If the time difference is less than or equal to the rolling window size, return the frequency and monetary
            if time_diff <= window_size:
                return row["Frequency"], row["Monetary"]
            else:
                return 0, 0  # If not in the window, return 0 for both

        # Apply compute_fm to each row, get the frequency and monetary average for each row
        rfm_fm[["Frequency", "MonetaryAvg"]] = rfm_fm.apply(compute_fm, axis=1, result_type='expand')
        rfm_fm = rfm_fm.drop(columns=["Monetary"])  # Drop the Monetary column

        rfm_fm = rfm_fm.groupby(id_col).apply(
            lambda group: pd.Series({
            "Frequency": group["Frequency"].sum(),
            "MonetaryAvg": (group["MonetaryAvg"] * group["Frequency"]).sum() / group["Frequency"].sum() if group["Frequency"].sum() > 0 else 0
            }), include_groups=False
        ).reset_index()

        rfm_final = pd.merge(rfm_r, rfm_fm, on=id_col, how="inner") 
        # Apply merging for the final RFM tables for each ***TIMEPERIOD*** not for each DONOR


        return rfm_final




    print(f"Initiating the calculation of RFM values at each Time Period for all the donors who have donated at least once up to that point...")
    print("Please wait... This may take a while depending on the sizes of the datasets and the time-steps.\n")
    # Create a dictionary to store RFM DataFrames for each Time-Step (ts)
    rfm_dfs = {}

    # Iterate over each unique TimePeriod and create DataFrames
    for i, ts in enumerate(unique_tp):
        print(f"Calculating the RFM values at Time Period {ts}...\n")
        rfm = calculate_RFM(tp_dfs[str(ts)])  # Calculate RFM for the current ts
        rfm_dfs[str(ts+1)] = rfm
        rfm = None  # Clear the variable rfm after each iteration


    # Save the RFM DataFrames to the "files" folder
    rfm_dfs_path = os.path.join(custom_folder_path, "rfm_dfs.pkl")
    with open(rfm_dfs_path, "wb") as f:
        pickle.dump(rfm_dfs, f)

    print(f"RFM DataFrames for each {'month' if time_step == 'M' else 'quarter' if time_step == 'Q' else 'year'} saved successfully!\n")

def install_packages():
    import sys
    import subprocess

    required_packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "scikit-optimize"
    ]

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing missing package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
