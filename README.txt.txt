Implementation of the pape: "Graph-based Non-Euclidean Mixture Model for Social Network Embedding"

Datasets: 
- BlogCatalog: https://dl.acm.org/doi/10.1145/3178876.3186102
- LiveJournal: https://snap.stanford.edu/data/com-LiveJournal.html
- Friendster: https://snap.stanford.edu/data/com-Friendster.html
- Wikipedia Clickstream: https://dl.acm.org/doi/10.1145/3178876.3186102
- Wikipedia Hyperlink: https://dl.acm.org/doi/10.1145/3178876.3186102
- Facebook: https://snap.stanford.edu/data/ego-Gplus.html
- Google+: https://snap.stanford.edu/data/ego-Gplus.html

-------------------

For the NMM and RaRE models: 

Usage:
javac *.java
java Main <options>
or

java Main -help
to show the help command below.

for NMM (main_new/Main.java) and RaRE (main_original/Main.java)

options:
-k: dimension of embedding; 2 by default
-data: the location of data file
-weighted: whether the network is weighted (0: unweighted; other: weighted); 0 by default
-out: the directory of output files; "." (current directory) by default
-lr: initial stepsize; 0.05 by default
-lambda_r: coefficient of dr (first component), must be positive; 1.0 by default
-lambda_z: coefficient of dz (second component), must be positive; 1.0 by default
-lambda_0: bias term (third component), should be negative for sparse networks; -1.0 by default
-alpha_r: parameter for power law prior (on r), recommended value: 1.0~2.0; 1.5 by default
-reg_z: l2 regularization coefficient (on z); 1e-6 by default
-tp: fraction of nodes for training, must be between 0 and 1; 0.9 by default
-tolerance: stopping criterion on log-likelihood improvement (negative number for no tolerance); 1e-6 by default
-nsw: negative sample weight, must be positive; 5 by default
-iter: value of maximum edges to be sampled (in thousands); 10000 thousand by default
-verbose: whether see verbose output or not (0: show limited outputs; other: show all outputs); 1 by default
-h or -help: show this help command
The -data argument is necessary.

For NMM-GNN: 
navigate to graph_vae/baselines/graphvae/model_baseline.py (contains implementation of NMM-GNN) and used by train script