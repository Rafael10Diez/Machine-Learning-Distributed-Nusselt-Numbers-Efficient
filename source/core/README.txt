./source/core
    a) Generic numerical routines to run a Machine Learning study.


./source/core/external.py
    a) Internal file importing functions from the parent ./all_interp directory.
        - Imported functions include the HeightFunction, log-file printer, etc.


./source/core/gather_data.py
    a) Routines to import data for the interpolated wall forces and heat transfer rates stored in the sub-folders of ./all_interp/data/output.


./source/core/io_access_ml.py
    a) Functions to print data into the ./machine_learning/data/output folder.


./source/core/loaders.py
    a) Functions to create the typical `data_loaders` found in PyTorch.

./source/core/ml_runner.py
    a) Main routine handling the Machine Learning studies. 
        - Input parameters are read
        - A Neural Network architecture is initialized and trained according to the configuration given.
        - The final prediction margins are established.
