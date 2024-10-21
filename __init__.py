import os

# generate data directory in the project root directory
project_root = os.path.dirname(__file__)
data_dir = os.path.join(project_root, 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# generate plot data directory
plot_data_dir = os.path.join(data_dir, 'plots')
if not os.path.exists(plot_data_dir):
    os.makedirs(plot_data_dir)

# generate map data directory
map_data_dir = os.path.join(data_dir, 'map')
if not os.path.exists(map_data_dir):
    os.makedirs(map_data_dir)

# get configuration data directory
params_data_dir = os.path.join(data_dir, 'params')
if not os.path.exists(params_data_dir):
    os.makedirs(params_data_dir)

# get samples data directory
samples_data_dir = os.path.join(data_dir, 'samples')
if not os.path.exists(samples_data_dir):
    os.makedirs(samples_data_dir)

models_data_dir = os.path.join(data_dir, 'models')
if not os.path.exists(models_data_dir):
    os.makedirs(models_data_dir)