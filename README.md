# DVMnet
Dual-view multitask learning for predicting lncRNA and miRNA related properties and interactions using improved de bruijn graph and heterogeneous graph

# Requirements
python                    3.8.10
numpy                     1.22.4   
pandas                    2.0.3 
scikit-learn              1.3.0    
torch                     1.11.0+cu113 
torchvision               0.12.0+cu113  

# Usage
If you use the default dataset, simply running the main.py file will suffice.

If you want to construct an improved de Bruijn graph on your own, there is commented code in the main.py file. By running the following code, you can construct the improved de Bruijn graph for this dataset. If you wish to construct an improved de Bruijn graph using your own dataset, you can also adopt this strategy.
de_vectors_lnc = de_bruijn_graph(lnc_seq,unique_lnc,3)
de_vectors_lnc = de_bruijn_graph(mi_seq,unique_mi,2)

Thank you and enjoy the tool!
