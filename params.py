axis_range = 10.0
im_height = 64
im_width = 64

# Data parameters
max_num_base_obstacle_maps = 8
max_index_base_obstacle_maps = 29
max_num_additional_obstacles = 19

lambda_scale = 0.15

# Training parameters
batch_size = 16
num_epochs = 3
num_batches_in_epoch = 8000
test_set_size = 125

# Inference parameters
network_path = ".\\models\\current_best_4\\test_model.pt"