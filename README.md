# balloon-data_analysis
codes for analyzing B-ICI balloon data

Use typhon package for ginding equilibrium vapor pressure of water and ice
https://www.radiativetransfer.org/misc/typhon/doc/index.html

Convert the PTU and the ETAG file from esrange. 
Use 'plots.py' to create the plot, and 'ptu_data.py' to convert the ptu file. 
The ETAG file might have to be tailored bz each launch. The code expects a specific form of the code, and that can varies with each launch. Check if the format looks exactly like how the program expects it.


### Using the models
In order to use the models, create a virtual environment (with conda).

## Segmentation model
Clone the git-repo to ".../data_analysis_tutorial/."

https://github.com/Omnitok/snow_crystal_segmentation

Use the dockerfile for installing the dependencies "*/data_analysis_tutorial/snow_crystal_segmentation/model/docker" "docker build -t cuda-tensorflow ."

The model is in snow_crystal_segmentation folder. Already pretrained, currently using model #m232.

Run it with "sudo ./run_inference.sh". Here you can change the input-output directories.

On "run_inference.py" One can set the detection limit for the model, and also can tweak the particle properties that the model will produce.

## Classification model
python=3.9
install the dependencies with "pip install -r requirements.txt"
The model is in the folder "classification".
Pretrained, but ideally with every measurements the hand/automaticly classified particles are fed back to the training, to train a new model.

Run it with "classify.py" where the input/oputput folders can be set.
