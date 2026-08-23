# Permuting a sparse matrix involves reordering its rows and columns to reveal structural properties, 
# such as a Block Upper Triangular Form (BTF). This transformation isolates independent 
# subsystems, which drastically accelerates linear solvers.The easiest and most efficient way 
# to achieve this in Python is by combining the Dulmage–Mendelsohn decomposition with a 
# Strongly Connected Components (SCC) refinement, a pipeline natively exposed via the 
# scipy.sparse.csgraph.structural_rank utility or the workhorse functions in scipy.sparse.
# Python Implementation (Complete Matrix Permutation)
# The script below takes a sparse matrix, computes its structural row and column permutations, 
# and rearranges the matrix so that the structural blocks sit neatly along the diagonal.python


# MechanismMaximum Matching: The algorithm shifts rows so that every entry along 
# the main diagonal A_{i,i} is non-zero. This maps the linear equations directly to 
# their corresponding variables.Directed Graph Construction: By treating the zero-free diagonal 
# matrix as a directed graph adjacency matrix, a non-zero value at A_{i,j} indicates that equation 
# i depends on variable j.

# Strongly Connected Components (SCC): Subsystems that rely mutually on each other form strongly 
# connected cycles. SCC algorithms group these tightly bound variables and equations together.

# Topological Block Sort: Sorting the SCCs ensures that if block B depends on block A, 
# block A is solved first. Visually, this shifts all remaining off-diagonal elements into the 
# upper triangular half, leaving the lower half completely clean (0).

# Visualizing Sparse Structural TransformationsTo verify that your permutation was successful, 
# look at the structural sparsity pattern (spy plot) of the matrix before and after.

# Before (Left): Non-zero elements are scattered randomly across the coordinates. 
# A solver must process the entire matrix at once.

# After (Right): Non-zero elements cluster tightly into isolated squares along the diagonal. 
# The solver can compute the bottom-right block first, feed its outputs upward, 
# and solve the remaining independent blocks sequentially.

import numpy as np
from scipy.sparse import csr_matrix, csgraph
from scipy.sparse.csgraph import maximum_bipartite_matching


def permute_sparse_matrix(matrix):
    """
    Permutes a sparse matrix into a Block Upper Triangular Form (BTF)
    using maximum matching and strongly connected components.
    """
    # Ensure matrix is in Compressed Sparse Row format
    A = csr_matrix(matrix)
    num_rows, num_cols = A.shape
   
    # Step 1: Compute Maximum Bipartite Matching to ensure a zero-free diagonal
    # row_perm maps new row positions to original row indices
    row_perm = csgraph.maximum_bipartite_matching(A, perm_type='row')
   
    # Check if a perfect matching exists (matrix must have full structural rank)
    if -1 in row_perm or num_rows != num_cols:
        # If asymmetric or rank-deficient, we fallback to structural_rank analysis
        # For this demonstration, we focus on invertible square blocks
        raise ValueError("Matrix must be square and have full structural rank for pure BTF.")
       
    # Apply row permutation to get a zero-free diagonal
    A_matched = A[row_perm, :]
   
    # Step 2: Find Strongly Connected Components (SCC) on the matched graph
    # Treating the zero-free diagonal matrix as an adjacency matrix
    n_components, labels = csgraph.connected_components(
        csgraph=A_matched, directed=True, connection='strong'
    )
   
    # Step 3: Sort components to achieve an upper triangular block layout
    # Group row/col indices by their SCC label ID
    component_order = np.argsort(labels)
   
    # Final permutation vectors
    # Combined row permutation: matching perm followed by component reordering
    final_row_perm = row_perm[component_order]
    final_col_perm = component_order
   
    # Step 4: Permute the original matrix
    A_permuted = A[final_row_perm, :][:, final_col_perm]
   
    return A_permuted, final_row_perm, final_col_perm

def dulmage_mendelsohn_coarse(matrix_or_adj):
    """
    Computes the coarse Dulmage-Mendelsohn decomposition.
    Assumes rows = U, columns = V of a bipartite graph.
    
    The decomposition can be split into two main structural resolutions:
    1. Coarse Decomposition
    Find a Maximum Matching: Use an algorithm like Hopcroft–Karp to find a maximum cardinality matching in the bipartite graph.
    Breadth-First Search (BFS) from Unmatched Nodes:
    Find all vertices reachable via alternating paths starting from unmatched column vertices. This forms the under-determined block.
    Find all vertices reachable via alternating paths starting from unmatched row vertices. This forms the over-determined block.
    Isolate the Remainder: Any remaining matched row and column vertices that were not reached form the well-determined (square) block.
    
    2. Fine Decomposition
    The square/well-determined block can be further refined:
    Direct the matched edges in one direction and unmatched edges in the opposite direction to build a directed graph.
    Run a Strongly Connected Components (SCC) algorithm (like Tarjan's or Kosaraju's).
    Each individual component forms an independent, irreducible square block.
    """
    graph = csr_matrix(matrix_or_adj)
    num_rows, num_cols = graph.shape
    
    # Step 1: Find maximum bipartite matching
    # row_to_col[i] = j means row i matches with column j. -1 means unmatched.
    row_to_col = maximum_bipartite_matching(graph, perm_type='column')
    
    # Track column to row matches
    col_to_row = np.full(num_cols, -1, dtype=int)
    for r, c in enumerate(row_to_col):
        if c != -1:
            col_to_row[c] = r

    # Step 2: BFS for Under-determined block (Start at unmatched columns)
    under_rows = set()
    under_cols = set(c for c in range(num_cols) if col_to_row[c] == -1)
    queue = list(under_cols)
    
    while queue:
        curr_col = queue.pop(0)
        # Find rows connected to this column
        for r in graph.getcol(curr_col).indices:
            if r not in under_rows:
                under_rows.add(r)
                # If this row is matched, its matched column joins the under-determined block
                matched_c = row_to_col[r]
                if matched_c != -1 and matched_c not in under_cols:
                    under_cols.add(matched_c)
                    queue.append(matched_c)

    # Step 3: BFS for Over-determined block (Start at unmatched rows)
    over_cols = set()
    over_rows = set(r for r in range(num_rows) if row_to_col[r] == -1)
    queue = list(over_rows)
    
    while queue:
        curr_row = queue.pop(0)
        # Find columns connected to this row
        for c in graph.getrow(curr_row).indices:
            if c not in over_cols:
                over_cols.add(c)
                # If this column is matched, its matched row joins the over-determined block
                matched_r = col_to_row[c]
                if matched_r != -1 and matched_r not in over_rows:
                    over_rows.add(matched_r)
                    queue.append(matched_r)

    # Step 4: Well-determined (Square) block is everything else
    square_rows = set(range(num_rows)) - under_rows - over_rows
    square_cols = set(range(num_cols)) - under_cols - over_cols

    return {
        "under-determined": {"rows": list(under_rows), "cols": list(under_cols)},
        "well-determined": {"rows": list(square_rows), "cols": list(square_cols)},
        "over-determined": {"rows": list(over_rows), "cols": list(over_cols)}
    }

# --- Example Usage ---
# Construct a 5x5 sparse matrix with distinct structural blocks
# Blocks are inter-dependent but can be isolated
data = np.array([
    [1, 0, 0, 2, 0],
    [0, 3, 4, 0, 0],
    [0, 5, 6, 0, 0],
    [0, 0, 0, 7, 8],
    [0, 0, 0, 0, 9]
])

A = csr_matrix(data)
A_btf, r_perm, c_perm = permute_sparse_matrix(A)

print("Original Row Order:", list(range(A.shape[0])))
print("Permuted Row Order:", list(r_perm))
print("Permuted Col Order:", list(c_perm))
print("\nPermuted Matrix Matrix Structure:\n", A_btf.toarray())
print()

# --- Example System ---
# 3 equations (rows), 4 variables (columns)
# Eq 0: x0, x1
# Eq 1: x1
# Eq 2: x2, x3
incidence_matrix = np.array([
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 1]
])

blocks = dulmage_mendelsohn_coarse(incidence_matrix)
for block_name, items in blocks.items():
    print(f"{block_name.capitalize()}: Rows {items['rows']}, Cols {items['cols']}")
    
    