import time
import os
from Preprocessing_transactions import process_transactions, install_packages
from donor_dynamics import process_RFM_per_tp, train_test_RFM_split
from dataset_scaler import scale_dataset
from windows_processing import window_construction, window_construction_deploy
import json
from datetime import datetime

# Install required packages for the entire code package
# install_packages()


### CONFIGURATION: Choose time-step granularity ('M' = Month, 'Q' = Quarter, 'Y' = Year)
time_step = "M"  # Use "M" for months, "Q" for quarters or "Y" for fiscal years

transaction_dataset_name = "UPMC_transactional.csv"
promo_dataset_name = "UPMC_promo.csv"

filter_unique_ids = 1 # Filter out donors until only {filter_unique_ids * 100}% of the unique IDs are retained
cut_off_date  = "2025-06-01" # Filter out transactions from the start of that time period onwards
test_size = 0.2 # Proportion of the dataset to include in the test split
model_window_size = 24 # Number of time periods to include in each window (For training and testing and deployment of the model)
exclude_one_time_donors = 0 # Exclude one-time donors from the dataset (1 = Exclude, 0 = Include)
decay_coefficient = 0.9 # Decay coefficient for the v variable (cumulative exposure). Every month, the previous month's cumulative exposure gets multiplied by this coefficient.

### COLUMN NAMES:
id_col = "constituent_id"  # Column name for unique donor IDs in the datasets e.g. "Account Case Safe ID" for CAFB dataset
tran_date_col = "date" # Column name for transaction dates in the transactions dataset e.g. "Close Date" for CAFB dataset
promo_date_col = "date" # Column name for promo dates in the promos dataset e.g. "Created Date" for CAFB dataset
amount_col = "amount"  # Column name for transaction amounts in the transactions dataset e.g. "Amount" for CAFB dataset
donation_type_col = "type"  # Column name for the type of donation in the transactions dataset (recurring or one-time) e.g. "Campaign Name" for CAFB dataset


### FILTERING OUT RECURRING DONORS
"""
Subscribers or recurring or regular donors are excluded based on the donation_type_col column in the transactions dataset for each donor using the "pattern", "match_pattern", "exclude_nans" variables.
- pattern: regex or string pattern to match
- match_pattern: if True, exclude rows matching the pattern; if False, exclude rows NOT matching the pattern
- exclude_nans: if True, exclude NaN rows regardless of pattern; if False, do not exclude NaNs regardless of pattern

Example:
Exclude rows matching the pattern, and also exclude NaNs. This pattern will exclude the donors that have any of these phrases in the donation_type_col as well as NaN values:
pattern = "friends of the Food bank|monthly|friend of the Food bank|August 2021 Sustainer Blitz Email Appeal"
match_pattern=True
exclude_nans=True
"""
pattern = "Donation"
match_pattern=False  # If True, exclude rows matching the pattern; if False, exclude rows NOT matching the pattern
exclude_nans = True  # If True, exclude NaN rows regardless of pattern; if False, do not exclude NaNs regardless of pattern





"""
***
The rest of the code processes the datasets, trains the model, and prepares the data for deployment. There is no need to modify the code below this point unless you want to change the processing logic or parameters.
***
"""
# Get current time as a string, e.g., "2024-06-20_15-30-45"
current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Files will be saved to:
folder_name = f"{filter_unique_ids*100}%_retained_{cut_off_date}_cut_off_date_{'Excluded' if exclude_one_time_donors == 1 else 'Included'}_one_time_donors_{model_window_size}_window_size_{current_time_str}"

current_dir = os.path.dirname(__file__)  # Get the current script's directory
files_dir = os.path.join(current_dir, "Files")  # Path to the "Files" directory
custom_folder_path = os.path.join(files_dir, folder_name)  # Path to the custom folder
os.makedirs(custom_folder_path, exist_ok=True)  # Create the custom folder if it doesn't exist
# Validate custom_folder_path
if not os.path.isdir(custom_folder_path):
    raise ValueError(f"Invalid directory: {custom_folder_path}")
# Create the "Deployment Files" folder inside the custom_folder_path if it doesn't exist
deployment_folder = os.path.join(custom_folder_path, "Deployment Files")
os.makedirs(deployment_folder, exist_ok=True
            )
# Incrementing the cut-off date by one month for deployment
# Convert the string to a datetime object
cut_off_date_dt = datetime.strptime(cut_off_date, "%Y-%m-%d")
# Calculate the start of the next month
if cut_off_date_dt.month == 12:  # If it's December, move to January of the next year
    next_month_start = cut_off_date_dt.replace(year=cut_off_date_dt.year + 1, month=1, day=1)
else:  # Otherwise, move to the first day of the next month
    next_month_start = cut_off_date_dt.replace(month=cut_off_date_dt.month + 1, day=1)
# Convert back to string
cut_off_date = next_month_start.strftime("%Y-%m-%d")

# Saving the configuration for use later in deployment
config = {
    "time_step": time_step,
    "transaction_dataset_name": transaction_dataset_name,
    "promo_dataset_name": promo_dataset_name,
    "filter_unique_ids": filter_unique_ids,
    "cut_off_date": cut_off_date,
    "test_size": test_size,
    "model_window_size": model_window_size,
    "exclude_one_time_donors": exclude_one_time_donors,
    "decay_coefficient": decay_coefficient,
    "id_col": id_col,
    "tran_date_col": tran_date_col,
    "promo_date_col": promo_date_col,
    "amount_col": amount_col,
    "donation_type_col": donation_type_col,
    "pattern": pattern,
    "match_pattern": match_pattern,
    "exclude_nans": exclude_nans
}

# Save the configuration to a JSON file
config_file_path = os.path.join(custom_folder_path, "config.json")
with open(config_file_path, "w") as f:
    json.dump(config, f, indent=4)

"""

IMPORTANT: Choose the cut-off date in the format "YYYY-MM-DD" and in a way that leaves out at least one time period
available in the original transactions dataset. 

This is done due to the nature of the model proposed by PRASAD A. NAIK and NANDA PIERSMA in 
"UNDERSTANDING THE ROLE OF MARKETING COMMUNICATIONS IN DIRECT MARKETING".

The model requires the calculation of the RFM variables for the preceding time period, which is why the cut-off date
should be set to exclude the last time period in the dataset. This is to ensure that the every set of RFM variables
have a "y" value associated with them.

"""
start_time = time.time()

process_transactions(time_step, filter_unique_ids, cut_off_date, transaction_dataset_name, custom_folder_path, exclude_one_time_donors)
process_RFM_per_tp(time_step, cut_off_date, promo_dataset_name, custom_folder_path, decay_coefficient)
train_test_RFM_split(custom_folder_path, test_size)
scale_dataset(custom_folder_path)
window_construction(custom_folder_path, model_window_size)
window_construction_deploy(custom_folder_path, model_window_size)

end_time = time.time()

print(f"Total time taken: {(end_time - start_time)/60:.2f} mins.")

# Save the custom_folder_path to a .txt file
path_file = os.path.join(files_dir, "custom_folder_path.txt")
with open(path_file, "w") as f:
    f.write(custom_folder_path)