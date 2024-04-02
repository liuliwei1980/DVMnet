
Dual-view multitask learning for predicting lncRNA and miRNA related properties and interactions using improved de bruijn graph and heterogeneous graph

# data：
mirna_lncrna_interaction.csv
di.csv
sub.csv
index_value.csv
node_link.csv

# Requirements
python                    3.8.10
numpy                     1.22.4   
pandas                    2.0.3 
scikit-learn              1.3.0    
torch                     1.11.0+cu113 
torchvision               0.12.0+cu113  

# Usage
If you use the default dataset, simply running the main.py file will suffice.

If you are using your own dataset, first call the de_bruijn_graph method in de_graph.py from within the main.py file to obtain the improved De Bruijn graph, and save it as a pkl file.

Thank you and enjoy the tool!
# DVMnet
