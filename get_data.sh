#!/usr/bin/env bash

export PROJ_ROOT=`pwd`
if [ ! -d "Data" ]; then
	mkdir Data
	pushd $PROJ_ROOT/Data
	wget http://snap.stanford.edu/data/amazon/productGraph/metadata.json.gz
	gunzip metadata.json.gz
	wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry_5.json.gz
	gunzip reviews_Clothing_Shoes_and_Jewelry_5.json.gz
	mkdir images
	mkdir Resize_images_50
	popd 
fi
pushd $PROJ_ROOT/Code/Data\ Handler
python extract_item_ids.py
python user_item_pairs.py
python extract_imageurls.py
python collect_images.py
python resize_images.py
popd