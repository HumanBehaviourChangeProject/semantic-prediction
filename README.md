# HBCP Semantic Prediction
The repository for the HBCP semantic-enhanced prediction system. 

We aim to develop an interpretable system for prediction of outcomes from behaviour change interventions using the Behaviour Change Intervention Ontology and a corpus of annotated literature. 

# How to run the rule prediction

1. Generate the dataset:
`python dataprocessing/dataprocessing.py`
  This should create a pickled version of the dataset in `data/hbcp_gen.pkl`

2. Run the model:
  `python rulenn/rule_nn.py data/hbcp_gen.pkl`
  
  The rules that have been trained should then be available under `results/rules.txt`. Rules are separated by semicolons. The last element of each rule is the rule weight.

# Data cleaning

The mechanisms used for data cleaning are documented in [docs/cleaner.md](docs/cleaner.md)

