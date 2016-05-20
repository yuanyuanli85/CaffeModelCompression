import kmeans_clustering as kc
import numpy as np

init_centro = np.array([1.0,2.0,3.5], dtype=np.float32)
centro = np.empty(init_centro.shape, dtype=np.float32)

nodes_list = np.array([1.1,3.3,1.9,3.4,0.8,2.1,2.7,1.8], dtype=np.float32)
label = np.empty(nodes_list.shape, dtype=np.int32)

print "nodes: ", nodes_list
print "init centros: ", init_centro
kc.kmeans_cluster(label, centro, nodes_list, nodes_list.size, 1, init_centro.size, init_centro, 5)
print "label: ", label
print "centros: ", centro
