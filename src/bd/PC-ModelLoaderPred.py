import tensorflow as tf
import prettytensor as pt
import numpy as np
 
def import_model(_num_timesteps, _grid_size, _batch_size):
    global num_timesteps
    global grid_size
    global batch_size
    num_timesteps = _num_timesteps
    grid_size = _grid_size
    batch_size = _batch_size
    return network()
 
def fc_layers(input_tensor, size):
    return (pt.wrap(input_tensor).
            fully_connected(256, name='common_fc1').
            fully_connected(size*size, activation_fn=tf.sigmoid, name='common_fc2').
            reshape([size, size])).tensor
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left, n_features)
		split(node['left'], max_depth, min_size, n_features, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right, n_features)
		split(node['right'], max_depth, min_size, n_features, depth+1)
 
# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
	root = get_split(train, n_features)
	split(root, max_depth, min_size, n_features, 1)
	return root

def network_conflict(input_tensor):    
    return (pt.wrap(input_tensor).
            conv2d(3, 2, stride=1).
            conv2d(5, 5, stride=1).
            flatten().
            fully_connected(128, activation_fn=None, name='conflict_fc1')).tensor
 
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']
 
       
def network(): 
    dim_0, dim_1, dim_2 = grid_size
    gt = tf.placeholder(tf.float32, [dim_0, dim_1]) 
    conflict_grids = tf.placeholder(tf.float32, [num_timesteps, dim_0, dim_1, dim_2])
    mask = tf.placeholder(tf.float32, [dim_0, dim_1])
    #poverty_grid = tf.placeholder(tf.float32, [1, dim_0, dim_1])
 
    assert(num_timesteps > 1)
    with tf.variable_scope("model") as scope:
        with pt.defaults_scope(activation_fn=tf.nn.relu,
                               batch_normalize=True,
                               learned_moments_update_rate=0.0003,
                               variance_epsilon=0.001,
                               scale_after_normalization=True):
            enc_conflicts = network_conflict(conflict_grids)
    
    mean_conflict = tf.reduce_mean(enc_conflicts, 0)
    mean_conflict = tf.reshape(mean_conflict, [1, 128])
    '''
    with tf.variable_scope("model") as scope:
        with pt.defaults_scope(activation_fn=tf.nn.relu,
                               batch_normalize=True,
                               learned_moments_update_rate=0.0003,
                               variance_epsilon=0.001,
                               scale_after_normalization=True):
            enc_poverty = network_poverty(poverty_grid)
 
    feats = tf.concat(0, [mean_conflict, enc_poverty])
    '''
    feats = mean_conflict
    pred = fc_layers(feats, dim_0)
 
    return conflict_grids, pred, gt, mask

def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
	trees = list()
	for i in range(n_trees):
		sample = subsample(train, sample_size)
		tree = build_tree(sample, max_depth, min_size, n_features)
		trees.append(tree)
	predictions = [bagging_predict(trees, row) for row in test]
	return(predictions)
 