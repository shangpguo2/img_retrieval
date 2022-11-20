# Image Retrieval
<p>This is a project from CS4185 in CityU</p>

## Requirements
```shell
pip install requirements.txt
```

## Getting Started
<p>The features of images in folder image.orig are generated before executing the program to speed up.</p>
<p>Thus, if new images are added to test, extract_file.py should be run first to get all the features.</p>

## Retrieve Image
After the features are settled, use the code below to execute the UI:
```shell
python main.py
```
The UI will pop up once execute successfully. Click the browse button for searching the template files folder (It will 
automatically go to the examples' folder). Next, click on a template you want to search on the left to retrieve the 
most similar image.
![Product Name Screen Shot][product-screenshot]



[product-screenshot]: UI.jpg