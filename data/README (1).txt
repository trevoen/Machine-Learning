Dataset used in "Multi-Task Deep Neural Networks for Ames Mutagenicity Prediction", 
authored by M.J. Martínez, M.V. Sabando, A.J. Soto, C. Roca, C. Requena-Triguero, N.E. Campillo, J.A. Páez and I. Ponzoni (2022)

The file "ames_mutagenicity_data.csv" contains all the compounds used for the development and evaluation of the models proposed in our paper. The compounds thereby listed were originally compiled by the Istituto Superiore di Sanita’ (https://www.iss.it/isstox) and result from a thorough preprocessing stage, consisting of different filtering, sanitization and canonicalization steps. For further details on the data preprocessing stage, please refer to our full paper: 10.26434/chemrxiv-2022-852tf 

The columns "TA98", "TA100", "TA102", "TA1535", "TA1537" correspond to the labels computed for each strain, whereas the column "Overall" corresponds to the ground-truth consensus label used for evaluating the final Ames mutagenicity prediction.

The column "Partition" displays three values: "Train", "Internal" and "External". During the grid search stage, we employed the "Train" and "Internal" partitions to perform the hyperparameter exploratory search, whereas during the five-fold cross validation stage, we combined the "Train" and "Internal" partitions in order to compute the five folds to be used in each trial. The "External" partition was only used in the final stage of our experimental design to evaluate the trained models. Please refer to our full paper for further details on our experimental workflow.

All the remaining columns correspond to molecular descriptors computed using Mordred.

Note: the three pairs of compounds in the dataset with Ids 440-1226, 318-1360, and 2352-2779 can be considered duplicates since one is the corresponding salt of the other, even when they have slight differences in several of their molecular descriptor values.