# Match photos in newspapers to photos on a reel
Install the required packages (in a virtual environment) using the following commands
```
python -m pip install -r requirements.txt
python main.py
```

## Usage
The script main.py accepts two command line arguments:
- input_folder, pointing to a folder containing:
    - A subfolder newspaper, containing:
        - .jpg images of newspaper scans
        - .xml files specifying image coordinates
    - A subfolder photo, containing:
        - .jpg images of (cropped) photos
    - A config.json, specifying the configuration of the data
- output_folder, pointing to a folder for the output, default is `output`. If this folder does not yet exist, it will be created. If it does exist, existing results will be overwritten. It will be populated with:
    - for every photograph from the list find N-best matches from the illustrations in newspapers 
    - verify whether match is 'true/false' (postprocessing)
    
See the examples in the tests folder for more details.
    
## Tests
There is a single test that performs the whole procedure for two different archives, using different configuration.
