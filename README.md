# MARL+GNN for Traffic Engineering Optimization
#### Code of paper "Is Machine Learning Ready for Traffic Engineering Optimization?"
#### To appear as a Main Conference Paper at IEEE ICNP 2021
#### Guillermo Bernárdez, José Suárez-Varela, Albert Lopez, Bo Wu, Shihan Xiao, Xiangle Cheng, Pere Barlet-Ros, Albert Cabellos-Aparicio.
#### Links to paper: [[ArXiv](https://arxiv.org/abs/2109.01445)]
#### Download datasets [here](https://bnn.upc.edu/download/marl-gnn-te_datasets/)


## Abstract
Traffic Engineering (TE) is a basic building block of the Internet. In this paper, we analyze whether modern Machine Learning (ML) methods are ready to be used for TE optimization. We address this open question through a comparative analysis between the state of the art in ML and the state of the art in TE. To this end, we first present a novel distributed system for TE that leverages the latest advancements in ML. Our system implements a novel architecture that combines Multi-Agent Reinforcement Learning (MARL) and Graph Neural Networks (GNN) to minimize network congestion. In our evaluation, we compare our MARL+GNN system with DEFO, a network optimizer based on Constraint Programming that represents the state of the art in TE. Our experimental results show that the proposed MARL+GNN solution achieves equivalent performance to DEFO in a wide variety of network scenarios including three real-world network topologies. At the same time, we show that MARL+GNN can achieve significant reductions in execution time (from the scale of minutes with DEFO to a few seconds with our solution).
