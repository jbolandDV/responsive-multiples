from __future__ import annotations
import os, pickle, json, time, warnings
import numpy as np
import pandas as pd
import joblib
from concurrent.futures import ThreadPoolExecutor

# tf.keras + TCN
import tensorflow as tf
from tensorflow.keras.models import load_model          
from tcn import TCN
from keras.layers import TFSMLayer
import keras

# Optuna
import optuna
from optuna.samplers import TPESampler
from optuna._experimental import ExperimentalWarning
warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

### CONFIGURATION:

num_tps_to_predict = 12
mean_type = "rms" # "rms" or "mean": "rms" for root mean square, "mean" for arithmetic mean. using this, you choose what metric is used for optimization and for the final results.
threshold = 7 # the sum of the optimized mailings for the future time-steps should not be bigger than threshold
lower_limit = 0 # number of mailings per month should not be below the lower_limit
upper_limit = 1 # number of mailings per month should not be above the upper_limit



# The rest of the code does not need modifications unless changes in the logic are needed.

current_dir = os.path.dirname(__file__)  # Get the current script's directory
files_dir = os.path.join(current_dir, "Files")  # Path to the "Files" directory

# Load the latest custom_folder_path from the .txt file
path_file = os.path.join(files_dir, "custom_folder_path.txt")
with open(path_file, "r") as f:
    custom_folder_path = f.read().strip()

config_file_path = os.path.join(custom_folder_path, "config.json")
with open(config_file_path, "r") as f:
    config = json.load(f)
decay_coefficient = config["decay_coefficient"]
model_window_size = config["model_window_size"]
cut_off_date = config["cut_off_date"]
id_col = config["id_col"]
exclude_one_time_donors = config["exclude_one_time_donors"]
one_off_ids = config["one_off_ids"]
cut_off_date = pd.Period(cut_off_date, freq="M")
cut_off_date = cut_off_date - 1 # decrement the cut-off date by one month for deployment (to account for incrementing it in the main_process_datasets.py code)



def build_lagged_labels(X):
    """
    X : (1, model_window_size, 8)  feature tensor
    y : (1, model_window_size)     binary labels

    Returns X with the y_lagged column added to it: (N, model_window_size, 9) for teacher forcing
    """
    N, T, _ = X.shape          # T should be model_window_size
    assert T == model_window_size, "Expecting model_window_size‑month windows"


    # Compute the condition for the new column: Recency and is_padded have to be zero for y_lagged in the same row to be 1
    new_column = ((X[:, :, 2] == 0) & (X[:, :, 7] == 0)).astype("float32")  # Shape: (N, model_window_size)

    # Add the new column to X
    X_updated = np.concatenate([X, new_column[..., np.newaxis]], axis=-1)  # Shape: (N, model_window_size, 8)
    return X_updated

def prob_function(_, X, y, num_u):
    y_true = y.flatten()
    valid   = np.where(~np.isnan(y_true))[0]

    # Ensure float32 & correct shape
    x_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    out = fast_predict(x_tensor)
    # Handle structured outputs from Keras/ConcreteFunction
    if isinstance(out, dict):
        out = next(iter(out.values()))
    elif isinstance(out, (list, tuple)):
        out = out[0]
    y_pred_probs = tf.convert_to_tensor(out).numpy().ravel()

    nan_idx = np.where(np.isnan(y_true))[0]#[1:]
    nan_idx = nan_idx[-num_u:]  # Get the last num_u indices where y_true is NaN. To make sure we are only predicting the future time periods. this line isn't really needed but I let it stay for redundancy


    # Taking the arithmetic or RMS mean of the predicted probabilities for the future time periods
    if mean_type == "mean":
        future_tps_mean = np.mean(y_pred_probs[nan_idx])
    elif mean_type == "rms":
        future_tps_mean = np.sqrt(np.mean(np.square(y_pred_probs[nan_idx])))
    else:
        raise ValueError("mean_type must be 'arithmetic' or 'rms'")
    
    acc_valid_rows  = (
        (y_pred_probs[valid] > 0.5).astype(int) == y_true[valid]
    ).mean() if valid.size else np.nan
    return future_tps_mean, acc_valid_rows, y_pred_probs[nan_idx]

def u_fcn(model, u_array, X_deploy, y_deploy, decay_coefficient):
    Xd = X_deploy.copy()                         # 5× faster, GIL‑free
    u_array = u_array.reshape(-1)
    num_u = u_array.reshape(-1).shape[0]  # number of u values to be estimated
    Xd[0, -num_u:, 5] = u_array

    for idx in range(-num_u, 0):
        if idx-1 < -num_u:
            continue   # skip the first u row because its v entry is already calculated.
        else:
            Xd[0, idx, 6] = decay_coefficient * Xd[0, idx-1, 6] + Xd[0, idx, 5]

    # ----- scaling BEFORE dropping the “u” column -----
    Xd[:, :, 0] = et1_scaler.transform(Xd[:, :, 0].reshape(-1, 1)).reshape(Xd[:, :, 0].shape)
    Xd[:, :, 1] = et2_scaler.transform(Xd[:, :, 1].reshape(-1, 1)).reshape(Xd[:, :, 1].shape)
    Xd[:, :, 2] = recency_scaler.transform(Xd[:, :, 2].reshape(-1, 1)).reshape(Xd[:, :, 2].shape)
    Xd[:, :, 3] = frequency_scaler.transform(Xd[:, :, 3].reshape(-1, 1)).reshape(Xd[:, :, 3].shape)
    Xd[:, :, 4] = money_ave_scaler.transform(Xd[:, :, 4].reshape(-1, 1)).reshape(Xd[:, :, 4].shape)
    Xd[:, :, 6] = v_scaler.transform(Xd[:, :, 6].reshape(-1, 1)).reshape(Xd[:, :, 6].shape)
    Xd[:, :, 7] = 0                          
    
    Xd = np.delete(Xd, 5, axis=2)                # drop “u” *after* transforms
    return prob_function(model, Xd, y_deploy, num_u)

def optimise_once(
    X_deploy: np.ndarray,
    y_deploy: np.ndarray,
    perm_len: int,
    *,
    n_trials: int = 5,
    n_jobs: int   = 16,
    startup: int  = 64,
    lower: int    = 0,
    upper: int    = 5,
    seed:  int    = 42,
):
    """
    Optimise `perm_len` integer controls for the specific X_deploy / y_deploy
    that the caller passes in.
    """
    # ── freeze copies so nothing mutates them accidentally ───────────
    X_const = np.asarray(X_deploy, dtype=np.float32).copy()
    y_const = np.asarray(y_deploy, dtype=np.float32).copy()
    X_const.setflags(write=False)
    y_const.setflags(write=False)

    # ── an objective that *explicitly* takes X, y via default params ──
    def objective(trial, X=X_const, y=y_const):
        u_vec = np.array(
            [trial.suggest_int(f"u{i}", lower, upper) for i in range(perm_len)],
            dtype=np.int64,
        )

        # Constraint: sum of inputs must be smaller or equal to the threshold
        if u_vec.sum() > threshold:
            return float("inf") # or a large number to penalize


        future_tps_mean, _, _ = u_fcn(
            model,
            u_vec.reshape(-1, 1),
            X,
            y,
            decay_coefficient,
        )
        return -future_tps_mean   # Optuna minimises

    # ── run Optuna ----------------------------------------------------
    sampler = TPESampler(n_startup_trials=startup, multivariate=True, seed=seed)
    study   = optuna.create_study(direction="minimize", sampler=sampler)

    with ThreadPoolExecutor(max_workers=n_jobs):
        # The objective receives the bound X, y; no globals involved.
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    best_u   = np.array([study.best_params[f"u{i}"] for i in range(perm_len)],
                        dtype=np.int64)
    best_val = -study.best_value
    return best_u, best_val, study

def objective(u_array):
    u_array = np.array(u_array).reshape(-1, 1)  # Reshape for u_fcn
    future_tps_mean, acc_valid_rows, y_pred_future = u_fcn(model, u_array, X_deploy, y_deploy, decay_coefficient)
    return future_tps_mean, acc_valid_rows, y_pred_future  

def build_infer_model(export_dir: str, model_window_size: int):
    """Wrap a SavedModel in a Keras Model for easy .__call__ usage."""
    layer = TFSMLayer(export_dir, call_endpoint="serving_default")
    x = keras.Input(shape=(model_window_size, 8), dtype=tf.float32, name="features")

    # Most exports with a single named input expect a dict with that name.
    # If yours was exported positionally, the fallback will handle it.
    try:
        y = layer({"features": x})
    except Exception:
        y = layer(x)

    return keras.Model(x, y, name="RFM_TCN_infer")

start_time = time.time()

classifier_folder = os.path.join(custom_folder_path, "Classifier Model")
classifier_model_path = os.path.join(classifier_folder, "RFM_TCN")

custom_objects = {
    "TCN": TCN,
    "Custom>TCN": TCN,
}

model = build_infer_model(classifier_model_path, model_window_size)

INPUT_SPEC = tf.TensorSpec(shape=[1, model_window_size, 8], dtype=tf.float32)   # (batch, time, feat)
# @tf.function(input_signature=[INPUT_SPEC], reduce_retracing=True)   # <‑‑ fixed signature
def _fast_predict(x):
    # force‑set the static shape inside the graph
    x.set_shape([1, model_window_size, 8])
    return model(x, training=False)           # returns logits/probs
fast_predict = tf.function(
    _fast_predict,
    input_signature=[INPUT_SPEC],
    reduce_retracing=True
).get_concrete_function()

print(f"TCN model loaded from {classifier_model_path}\n")

scalers_path = os.path.join(custom_folder_path, "dataset_scalers.pkl")
with open(scalers_path, "rb") as f:
    scalers = pickle.load(f)

et1_scaler = scalers["et1_scaler"]
et2_scaler = scalers["et2_scaler"]
recency_scaler = scalers["recency_scaler"]
frequency_scaler = scalers["frequency_scaler"]
money_ave_scaler = scalers["money_ave_scaler"]
v_scaler = scalers["v_scaler"]

print("\nLoading the deployment windows...\n")
print("This may take a while depending on the number of donors and the size of the windows.\n")
# Get the path to the "Deployment Files" folder inside the custom_folder_path 
deployment_folder = os.path.join(custom_folder_path, "Deployment Files")
deploy_windows_path = os.path.join(deployment_folder, "deploy_donor_windows.joblib")

deploy_donor_windows = joblib.load(deploy_windows_path)

# with open(deploy_windows_path, "rb") as f:
#     deploy_donor_windows = pickle.load(f)   

print("Starting deployment optimization...") 
i = 0
for donor in list(deploy_donor_windows.keys()):
    i+=1

    (print(f"Optimizing for donor #{i} out of {len(deploy_donor_windows.keys())} donors...") if i % 100 == 0 else None)

    for perm_len in deploy_donor_windows[donor].keys():
        if model_window_size - perm_len +1 != num_tps_to_predict: # to only solve the optimization problems for the permutations that are needed for deployment.
            continue
        else:
            real_len      = perm_len - 1 # to count and make the algorithm give predictions for the latest timeperiod available (with NaN y) towards the projected rows.
            projected_len = model_window_size - real_len           # optimise these rows.  
            
            donor_data = deploy_donor_windows[donor][perm_len][0]
            donor_label = deploy_donor_windows[donor][perm_len][1]
            donor_data = donor_data.drop(columns=["TimePeriod"], inplace=False)

            X_deploy = donor_data.to_numpy()  
            y_deploy = donor_label.to_numpy()    
            X_deploy = np.expand_dims(X_deploy, axis=0) # shape: (1, model_window_size, 8)
            y_deploy = np.expand_dims(y_deploy, axis=0) # shape: (1, model_window_size)

            X_deploy = build_lagged_labels(X_deploy)  
            y_deploy = y_deploy[..., np.newaxis]


            best_u, best_val, _ = optimise_once(X_deploy, y_deploy, projected_len, n_trials=50, n_jobs=16, lower=lower_limit, upper=upper_limit, startup=20)   # Tweak these variables to adjust the optimization precision and speed. upper: upper bound of u, startup: number of trials to start with. n_trials: total number of trials.

            # Getting the results for the default situations where no marketing is done
            default_val, _, y_pred_future_default = objective(np.zeros([projected_len,1]).reshape(-1).reshape(-1, 1))
            # Getting the results for the situations where marketing is optimal
            _, acc_valid_rows, y_pred_future_optimal = objective(best_u)

            deploy_donor_windows[donor][perm_len].append(best_u)
            deploy_donor_windows[donor][perm_len].append(best_val)
            deploy_donor_windows[donor][perm_len].append(default_val)
            deploy_donor_windows[donor][perm_len].append(acc_valid_rows)

            # Adding the last non-padded Recency entry to the final results:
            # Find the index of the last 0 in the is_padded column
            last_zero_idx = deploy_donor_windows[donor][perm_len][0][deploy_donor_windows[donor][perm_len][0]['is_padded'] == 0].index[-1]

            # Get the Recency value at that index
            recency_at_last_zero = deploy_donor_windows[donor][perm_len][0].loc[last_zero_idx, 'Recency']
            deploy_donor_windows[donor][perm_len].append(recency_at_last_zero)

            # Storing the optimal and default marketing strategy probabilities
            deploy_donor_windows[donor][perm_len].append(y_pred_future_default)
            deploy_donor_windows[donor][perm_len].append(y_pred_future_optimal)

            


print("\nDeployment optimization completed.\n")
print("Saving the results...\n")
# Create the "Results" folder inside the custom_folder_path if it doesn't exist
results_folder = os.path.join(custom_folder_path, "Results")
os.makedirs(results_folder, exist_ok=True)
final_results_path = os.path.join(results_folder, "deploy_donor_results.pkl")
# with open(final_results_path, "wb") as f:
#         pickle.dump(deploy_donor_windows, f)


def confidence_score(acc: float | np.floating,     # k / n  or  np.nan
                     n:   int,
                     w:   float = 0.4) -> float:
    """
    Compute confidence = (k+1)/(n+4) * (max(n,1)/10)**w
    where n is #real rows (0‑10), k is derived from acc.
    """
    if n == 0 or np.isnan(acc):
        k = 0
        n_eff = 1                # ensures non‑zero coverage term
    else:
        k = int(round(acc * n))
        n_eff = n

    smoothed_acc = (k + 1) / (n + 4)          # Bayesian smoothing (α=1, β=3)
    coverage     = (n_eff / (model_window_size-1)) ** w          # size emphasis (w = 0.4). (model_window_size-1) because at most (model_window_size-1) real rows will be present in a deployment RFM window.

    return float(smoothed_acc * coverage)



k = 0
j = 0
rows = []  # Collect rows as lists
for donor in list(deploy_donor_windows.keys()):
    j += 1
    (print(f"Saving the results for donor {j} out of {len(deploy_donor_windows.keys())} donors...") if j % 1000 == 0 else None)
    for perm_len in deploy_donor_windows[donor].keys():

        if model_window_size - perm_len +1 != num_tps_to_predict: # to only solve the optimization problems for the permutations that are needed for deployment.
            continue
        else:

            recency = deploy_donor_windows[donor][perm_len][6]
            y_pred_future_optimal = deploy_donor_windows[donor][perm_len][8]
            y_pred_future_default = deploy_donor_windows[donor][perm_len][7]
            acc_valid_rows = deploy_donor_windows[donor][perm_len][5]
            ip = deploy_donor_windows[donor][perm_len][4]
            op = deploy_donor_windows[donor][perm_len][3]
            best_u = deploy_donor_windows[donor][perm_len][2]
            best_u = best_u.tolist()
            best_u = [f"{val:.1f}" for val in best_u]
            probability_boost = op - ip
            
            ## Append the row as a list:

            # first we make a portion of the row that consists of the before-and-after probabilities
            y_pred_comparison_rows = []
            for idx in range(len(y_pred_future_default)):
                y_pred_comparison_rows.append(f"{y_pred_future_default[idx]:.10f}" + " ---> " + f"{y_pred_future_optimal[idx]:.10f}")

            # now make up the whole row as a list
            if mean_type == "mean":
                rows.append(
                    [donor, recency, confidence_score(acc=acc_valid_rows, n=perm_len - 1), 
                     f"{ip:.10f}", f"{op:.10f}", f"{probability_boost:.10f}", 
                     f"{np.sqrt(np.mean(np.square(y_pred_future_default))):.10f}",
                     f"{np.sqrt(np.mean(np.square(y_pred_future_optimal))):.10f}",
                     f"{np.sqrt(np.mean(np.square(y_pred_future_optimal)))-np.sqrt(np.mean(np.square(y_pred_future_default))):.10f}"] + best_u + y_pred_comparison_rows
                )
            elif mean_type == "rms":
                rows.append(
                    [donor, recency, confidence_score(acc=acc_valid_rows, n=perm_len - 1), 
                     f"{np.mean(y_pred_future_default):.10f}",
                     f"{np.mean(y_pred_future_optimal):.10f}",
                     f"{np.mean(y_pred_future_optimal)-np.mean(y_pred_future_default):.10f}",
                     f"{ip:.10f}", f"{op:.10f}", f"{probability_boost:.10f}"] + best_u + y_pred_comparison_rows
                )
            
            np.mean(y_pred_future_default)
    
donor_data_tps = [cut_off_date + i for i in range(1, num_tps_to_predict+1)] # to get the prediction timeline. returns a list that contains the time periods to make predictions for.
tps = donor_data_tps
tps = [str(period) for period in tps]
tps_comparison = [period_str + " (default and optimal probability comparison)" for period_str in tps]


df_columns = [id_col, "Latest Recency", "Confidence Score", "No-Marketing Probability Mean", "Optimized Probability Mean", "Probability Mean Boost", "No-Marketing Probability RMS", "Optimized Probability RMS", "Probability RMS Boost"] + tps + tps_comparison
main_df = pd.DataFrame(rows, columns=df_columns)
main_df = main_df.reset_index(drop=True)

variations = []  # List to store the variations of main_df

# Loop to create different variations of the optimization solutions
for i in range(10, 8+num_tps_to_predict+2):  # Columns 10 to the end column. 10 because the mailing strategy starts at the column index 9.
    # Filter rows where columns up to `i` are non-NaN and columns after `i` include NaN
    filtered_df = main_df[
        ~pd.to_numeric(main_df.iloc[:, i], errors='coerce').notna() &  # Column i is NOT a number
        main_df.iloc[:, 0:i].notna().all(axis=1)  # All columns up to `i` must be non-NaN
    ]

    # setting up sorting order based on mean_type
    if mean_type == "mean":
        sort_1, sort_2, sort_3 = 5, 1, 2 # Probability Mean Boost over Confidence Score over Latest Recency
    elif mean_type == "rms":
        sort_1, sort_2, sort_3 = 8, 1, 2 # Probability RMS Boost over Confidence Score over Latest Recency

    filtered_df = filtered_df.sort_values(
        by=[df_columns[sort_1], df_columns[sort_2], df_columns[sort_3]], 
        ascending=[False, True, False]
    )

    # # Drop columns from `i` onwards
    # filtered_df = filtered_df.iloc[:, :i]  # Keep only columns up to `i`
    variations.append(filtered_df)  # Add the filtered DataFrame to the list


# Save each variation to a separate CSV file
j = 1
for variation in variations:
    if j == 1:
        base_name = f"optimized_marketing_{j}_future_time_step"
    else:
        base_name = f"optimized_marketing_{j}_future_time_steps"

    variation_csv_path = os.path.join(results_folder, f"{base_name}_{mean_type}_metric")

    # Save all donors
    if variation.shape[0] != 0:
        os.makedirs(variation_csv_path, exist_ok=True)
        variation.to_csv(os.path.join(variation_csv_path, f"all_donors.csv"), index=False)

    if exclude_one_time_donors == 1:
        # Only non-one-off donors
        if variation.shape[0] != 0:
            os.makedirs(variation_csv_path, exist_ok=True)
            variation.to_csv(os.path.join(variation_csv_path, f"non_one_off_donors.csv"), index=False)

    elif exclude_one_time_donors == 0:
        variation_one_off = variation[variation[id_col].isin(one_off_ids)]
        variation_non_one_off = variation[~variation[id_col].isin(one_off_ids)]

        if variation_non_one_off.shape[0] != 0:
            os.makedirs(variation_csv_path, exist_ok=True)
            variation_non_one_off.to_csv(os.path.join(variation_csv_path, f"non_one_off_donors.csv"), index=False)
        if variation_one_off.shape[0] != 0:
            os.makedirs(variation_csv_path, exist_ok=True)
            variation_one_off.to_csv(os.path.join(variation_csv_path, f"one_off_donors.csv"), index=False)

    j += 1

# Replace NaN values with "-"
main_df = main_df.fillna("-")
# Sort main_df by "Probability Boost" (descending), latest recency, and then by "Confidence Score" (descending)
main_df = main_df.sort_values(
    by=[df_columns[sort_1], df_columns[sort_2], df_columns[sort_3]], 
    ascending=[False, True, False]
)


final_results_csv_path = os.path.join(results_folder, "optimized_marketing_all_future_time_steps.csv")
# Save the DataFrame to a CSV file
# main_df.to_csv(final_results_csv_path, index=False)

print(f"Optimized marketing tables saved to {results_folder}")



end_time = time.time()

print(f"Total time taken: {(end_time - start_time)/60:.2f} mins.")



