# This script implements Tarjan's algorithm with an explicit call stack. 
# It avoids recursion limits and can handle massive datasets directly. 

# Deadlock Detection: If any found SCC contains more than 1 node (or a single node with a self-loop), 
# you have a cyclic dependency, which implies a deadlock in lock-allocation graphs.
# Topological Sort with Cycles: Tarjan's algorithm naturally outputs SCCs in reverse topological order. 
# Reversing the result list gives you a clean topological sort of the components themselves.

def find_sccs_iterative(graph):
    """
    Finds Strongly Connected Components (SCCs) iteratively to completely avoid Python's recursion limit.
    """
    current_id = 0
    ids = {}
    lowlinks = {}
    on_stack = {}
    stack = []
    result = []

    # Main loop over all nodes to handle disconnected graph regions
    for start_node in graph:
        if start_node in ids:
            continue

        # Explicit call stack to mimic recursive DFS
        # Format: [node, neighbor_iterator_index]
        call_stack = [[start_node, 0]]

        while call_stack:
            curr_node, index = call_stack[-1]
            neighbors = graph.get(curr_node, [])

            # First time visiting this node
            if index == 0:
                ids[curr_node] = current_id
                lowlinks[curr_node] = current_id
                current_id += 1
                stack.append(curr_node)
                on_stack[curr_node] = True

            # Process the next neighbor
            if index < len(neighbors):
                neighbor = neighbors[index]
                # Increment the index for when we return to this node
                call_stack[-1][1] += 1

                if neighbor not in ids:
                    # Neighbor unvisited: push to call stack (simulate recursion)
                    call_stack.append([neighbor, 0])
                elif on_stack.get(neighbor, False):
                    # Neighbor on stack: update lowlink immediately
                    lowlinks[curr_node] = min(lowlinks[curr_node], ids[neighbor])
            
            else:
                # All neighbors processed; pop from call stack
                call_stack.pop()

                # If we have a parent frame, update the parent's lowlink (post-order callback)
                if call_stack:
                    parent_node = call_stack[-1][0]
                    lowlinks[parent_node] = min(lowlinks[parent_node], lowlinks[curr_node])

                # If this node is the root of an SCC, extract it
                if ids[curr_node] == lowlinks[curr_node]:
                    scc = []
                    while True:
                        top_node = stack.pop()
                        on_stack[top_node] = False
                        scc.append(top_node)
                        if top_node == curr_node:
                            break
                    result.append(scc)

    return result


# --- Verification Example ---
if __name__ == "__main__":
    # Directed graph with cycles
    # 0 -> 1 -> 2 -> 0 (cycle)
    # 2 -> 3
    # 3 -> 4 -> 3 (cycle)
    sample_graph = {0:[1],1:[2],2:[0,3],3:[4],4:[3]}
    
    print("SCCs:", find_sccs_iterative(sample_graph))
