# DonorVoice_Marketing_Optimizer

The order of running the scripts is as follows:
1) main_process_datasets.py
2) main_train_TCN.py
3) main_deployment.py

IMPORTANT: The scripts should be run sequentially and not in parallel as they are dependent on each other like items in a series.

In the main directory, a folder named "Datasets" should exist, which contains the transactional and promotional datasets. To specify which datasets are to be processed, you can adjust the "main_process_datasets.py" to operate on your desired datasets.

After all three scripts are executed successfully, you can find the final .csv results in the following directory: Main Directory / Files / [Simulation Details] / Results / [Optimization Details] /all_donors.csv
