# Kahn's Algorithm is a Breadth-First Search (BFS) technique used to find a topological sort of a Directed Acyclic Graph (DAG). 
# It acts by systematically removing nodes with zero in-degrees (nodes without any prerequisites or incoming edges). 

# Core Steps:
# Calculate In-Degrees: Track the number of incoming edges pointing to each node.
# Seed the Queue: Place all vertices with an in-degree of 0 into a processing queue.
# Reduce Dependencies: Pop a node from the queue, append it to your result list, and subtract 1 from the in-degrees of its neighbors.
# Iterate: If a neighbor's in-degree drops to 0, add it to the queue.
# Detect Cycles: If the final sorted list contains fewer elements than the total number of graph vertices, a cycle exists. 

from collections import deque, defaultdict

def kahn_topological_sort(num_vertices, edges):
    # Step 1: Initialize adjacency list and in-degree tracking
    adj_list = defaultdict(list)
    in_degree = {i: 0 for i in range(num_vertices)}
    
    # Build the graph and calculate in-degrees
    for u, v in edges:
        adj_list[u].append(v)
        in_degree[v] += 1
        
    # Step 3: Collect all nodes with 0 in-degree
    queue = deque([node for node in in_degree if in_degree[node] == 0])
    topo_order = []
    
    # Process the queue
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        
        # Decrease the in-degree of neighboring nodes
        for neighbor in adj_list[node]:
            in_degree[neighbor] -= 1
            # If in-degree becomes 0, add it to the queue
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                
    # Step 5: Check for cycles
    if len(topo_order) != num_vertices:
        raise ValueError("Graph contains a cycle! Topological sort impossible.")
        
    return topo_order

# --- Example Usage ---
# Graph definition: 5 nodes, dependencies as edge tuples (from -> to)
num_nodes = 5
dependency_edges = [(0, 1), (1, 2), (3, 2), (3, 4)]

result = kahn_topological_sort(num_nodes, dependency_edges)
print("Topological Sort Order:", result) 
# Possible Output: [0, 3, 1, 4, 2]

