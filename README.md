# Attentive-CF
Visually Explainable Recommendations

## Requirements
Python 2.7

Pytorch

## Data Collection
The data is taken from [here](http://jmcauley.ucsd.edu/data/amazon/). We have specifically worked with Clothing, Shoes and Jewelry data. Place the data.json and Dataitem_ids.txt inside a "Data" folder and place that folder beside the Code folder. Inside the "Data" folder, create 2 folders- "images" and "Resize_images_50". Next run extract_images.py, collect_images.py and resize_images.py in that order. 

## Code (Files to care about)
There are some files that don't have explanation as they are not directly involved with the project

### Data Preprocessing Files
* extract_imageurls.py - This script extracts imageurls into a file from the metadata available
* collect_images.py - This script downloads all those images from those urls into a folder
* resize_images.py - This script resizes them all to be of the same size
* user_item_pairs.py - This script extracts the user, items interaction from the metadata

### Core Model Files
* model.py - This file mainly consists of the model and helper funtions like training function
* constants.py - This file consists of environment variables 
* main.py - This file is the one that is to be run to learn the model
