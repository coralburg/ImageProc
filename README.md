# ImageProc By Coral Burg

## Project's goal
This project goal is to extract data from a form filled by handwritten Hebrew script.

## Introduction
Digital handwritten documents pose real challenges for automatic writer identification, keyword search-
ing, and indexing. Text line alignment, word segmentation and recognition and character segmentation
and recognition of document images are essential pre-processing operations for these automatizing
problems. Developing and testing such operation algorithms require labeled data. Hence this project
aims at developing a labeled text line alignment dataset, a labeled word dataset and a labeled character
dataset of handwritten Hebrew script.

## Input 
Input is a color image of handwritten filled form.

For example:
![Handwritten filled form](https://github.com/coralburg/ImageProc/blob/master/Example_input.jpg)

## Outputs
There are 2 different outputs. They are cropped text lines, cropped words and cropped characters. All
outputs will be saved in PAGE xml format.

For example:
[Output example](https://github.com/coralburg/ImageProc/blob/master/input_output/output/Scan_0001.xml)
