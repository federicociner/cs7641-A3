### CS7641 Assignment 3
### Federico Ciner (fciner3)
### Spring 2018

This project contains all of the necessary code and data to run the experiments used in CS7641 Assignment 3.

## Setup
1. Ensure you have Python 2.7 installed (any version above 2.7.10 will work) on your system, along with all standard libraries including the 'pip' and 'setuptools' packages.

2. Run 'pip install -r requirements.txt' to install the necessary Python libraries.

## Running the experiments

1. To perform data pre-processing run the main() functions in the 'data_preprocess.py.' The processed data sets will be written to the 'data/experiments' folder.

2. To run all the experiments, produce all results CSVs and plots, including the trained NN results, simply run the _run.sh_ script in the *src* folder. It will execute the various Python modules in the correct order and output all the relevant results.

## Credits and References
All implementations and experiment code for this assignment were taken from the Python scikit-learn library (http://scikit-learn.org) and the experiment code was adapted from Jonathan Tay's repository (https://github.com/JonathanTay/CS-7641-assignment-3).
