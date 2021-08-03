# DeepSpeech2 Prediction
This repository demos how to do prediction using DeepSpeech2.
The process followed by the one deployed on server uses `CNN_RNN.py` to perform both training and prediction (may be not).

This repository is just the simpler version of it, without any complexities of training procedure.

# Setup
- Install Conda environment.
- use the included `environment.yml` file to create the conda environment. 

**Note:** It may have some unnecessary packages as well, but those can be ignored for time-being.


# Usage
In `prediction.ipynb` file, change the directory path in `PROJECT_ROOT_DIR` appropriately and execute the notebook.

# Directory structure
## data
- Contains the .wav audio files.
- DeepSpeech2 training procedure requires a fixed directory, but that has been relaxed for prediction process

## ml-models
- The subdirectories contain the model (only weights) and preprocessing objects as well.

## src
- Contains custom files developed for DeepSpeech2

# Additional Info
I have used vscode, so few things may vary for the notebook environment.