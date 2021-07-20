# Anomaly Detecton

This is a series of python scripts and notebooks that cover anomaly detection.  

The classification folder shows how to train a classification algorithm on an imbalanced dataset.  This assumes that you have the labels within your dataset.  

The autoencoder folder shows how to train a feed forward autoencoder algorithm on a unlabled dataset.  The autoencoder learns from the trained data, where the autoencoder attempts to reproduce the input.  Next, the reconstruction error is used as a metric from anomalies.  High reconstruction error indicates possible anomalies.  
