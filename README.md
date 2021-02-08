# CAE-Ensemble

Code for the paper Time Series Outlier Detection with Diversity-Driven Convolutional Ensembles (under review)

Steps to obtain the main results:
 1. Run [Trainer.py](1-Trainer.py) over one of the datasets to train the ensemble members.
 2. Executing [Testing.py](2-Testing.py) over the trained models calculate the scores for each ensemble member.
 3. [Metrics.py](3-Metrics.py) calculate the Precision, Recall, F1, ROC-AUC, and PR-AUC metrics in CSV format.
 4. [ParallelSpot.py](4-ParallelSpot.py) calculate the automatic threshold given the scores of step #2.
 
 A running example will be available soon.
