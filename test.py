from datanetAPI import DatanetAPI

path = '/Users/gbg141/Documents/BNN-UPC/Projects/GNN-based_MARL_routing/datasets/gnnet_data_set_training'
intensity_values = [1400, 1600]

reader = DatanetAPI(path, intensity_values=intensity_values)
it = iter(reader)
for sample in it:
    routing = sample.get_routing_matrix()