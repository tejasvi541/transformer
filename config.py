from pathlib import Path

def get_config():
    # Returns a dictionary containing configuration settings for model training.
    # The configuration includes:
    # - batch_size: Number of samples per training batch.
    # - num_epochs: Number of training epochs.
    # - lr: Learning rate for the optimizer.
    # - seq_len: Maximum sequence length for input data.
    # - d_model: Dimension of the model (hidden layer size).
    # - datasource: Name of the data source being used (e.g., 'opus_books').
    # - lang_src: Source language code for the translation task (e.g., 'en' for English).
    # - lang_tgt: Target language code for the translation task (e.g., 'fr' for French).
    # - model_folder: Directory where model weights will be saved.
    # - model_basename: Base name used for saving model weights files.
    # - preload: Pretrained model path, if any, to load at the start (None means no preload).
    # - tokenizer_file: Filename pattern for saving/loading the tokenizer, using format string (e.g., 'tokenizer_en.json').
    # - experiment_name: Name of the folder where experiment results (e.g., logs) will be saved.

    return {
        "batch_size": 8,  # Number of training examples per batch.
        "num_epochs": 20,  # Total number of training epochs.
        "lr": 10**-4,  # Learning rate for model training.
        "seq_len": 512,  # Maximum length of input sequences for training.
        "d_model": 512,  # Dimension of the model (e.g., hidden layer size in the transformer).
        "datasource": 'opus_books',  # Source of the dataset being used.
        "lang_src": "en",  # Source language for translation (English).
        "lang_tgt": "fr",  # Target language for translation (French).
        "model_folder": "weights",  # Folder where model weights are saved.
        "model_basename": "tmodel_",  # Base name for model weights files.
        "preload": None,  # Path to a pre-trained model file (None if not loading).
        "tokenizer_file": "tokenizer_{0}.json",  # Filename template for the tokenizer (formatable with language code).
        "experiment_name": "runs/tmodel"  # Directory for saving experiment runs/logs.
    }

def get_weights_file_path(config, epoch: str):
    # Constructs the file path for saving or loading model weights for a given epoch.
    # Inputs:
    # - config: Dictionary containing the configuration settings (from get_config).
    # - epoch: String representing the specific epoch number for the model weights file.
    # Output:
    # - Returns the full path (as a string) to the model weights file for the given epoch.

    model_folder = f"{config['datasource']}_{config['model_folder']}"  # Create folder path using data source and model folder.
    model_filename = f"{config['model_basename']}{epoch}.pt"  # Create filename using base name and epoch number.
    return str(Path('.') / model_folder / model_filename)  # Return full path to the weights file.

def latest_weights_file_path(config):
    # Finds the latest model weights file (most recent by filename) from the weights folder.
    # Inputs:
    # - config: Dictionary containing the configuration settings (from get_config).
    # Output:
    # - Returns the full path (as a string) to the latest model weights file or None if no files are found.

    model_folder = f"{config['datasource']}_{config['model_folder']}"  # Create folder path using data source and model folder.
    model_filename = f"{config['model_basename']}*"  # Filename pattern to match all weight files with the given base name.
    weights_files = list(Path(model_folder).glob(model_filename))  # List all files in the folder matching the pattern.
    if len(weights_files) == 0:
        # If no weight files are found, return None.
        return None
    weights_files.sort()  # Sort the files by name to ensure chronological order.
    return str(weights_files[-1])  # Return the latest (last) file in the sorted list.
