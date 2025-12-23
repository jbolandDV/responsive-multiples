# RFM TCN Trainer (GPU)

Dockerized trainer that **does not include data**. At runtime, it mounts your local `Files/` folder and the code resolves the correct sub-folder using the **first line** of `Files/custom_folder_path.txt`.

Added docker-compose.yml, Dockerfile, .dockerignore, requirements.txt, and README_DOCKER.md

## Requirements
- Docker Desktop (or Docker Engine)
- NVIDIA drivers + NVIDIA Container Toolkit (for GPU)
- Your data on host at:
      
      Files/
  
        custom_folder_path.txt # first line ends with ...\YourRunFolder
      
        YourRunFolder/
      
          X_train.pkl
          y_train.pkl
          X_test.pkl
          y_test.pkl
          config.json
