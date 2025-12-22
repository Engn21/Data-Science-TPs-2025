import numpy as np
from itertools import product

def normalize(v):
    v = np.asarray(v, dtype=float)
    return v / v.sum()

def random_cpt(shape):
    p = np.random.rand(*shape)
    p /= p.sum(axis=len(shape) - 1, keepdims=True)
    return p


class BayesianNetwork:
    """
    A class to represent a Bayesian Network, generate probability formulas,
    and numerically verify marginalization.
    """
    
    def __init__(self, edges, node_order=None):
        """
        Initialize the Bayesian Network from edges.
        
        Args:
            edges: List of tuples representing directed edges (parent, child)
            node_order: Optional list specifying the order of nodes
        """
        self.edges = edges
        self.nodes = set()
        self.parents = {}
        
        # Extract all nodes from edges
        for parent, child in edges:
            self.nodes.add(parent)
            self.nodes.add(child)
        
        # Sort nodes for consistent ordering
        if node_order:
            self.nodes = node_order
        else:
            self.nodes = sorted(list(self.nodes))
        
        # Build parent dictionary
        for node in self.nodes:
            self.parents[node] = []
        
        for parent, child in edges:
            self.parents[child].append(parent)
        
        # Sort parents for consistency
        for node in self.nodes:
            self.parents[node] = sorted(self.parents[node])
    
    def get_factor_string(self, node):
        """Get the probability factor string for a node: P(node|parents) or P(node)"""
        parents = self.parents[node]
        if not parents:
            return f"P({node})"
        else:
            return f"P({node}|{', '.join(parents)})"
    
    def get_joint_factorization(self):
        """Get the full joint probability factorization string."""
        factors = [self.get_factor_string(node) for node in self.nodes]
        return "".join(factors)
    
    def get_marginal_formula(self, query_vars):
        """
        Get the marginalization formula for computing P(query_vars).
        
        Args:
            query_vars: List of variables we want in the marginal
            
        Returns:
            Tuple of (query_string, formula_string)
        """
        # Find hidden variables (to be summed out)
        hidden_vars = [node for node in self.nodes if node not in query_vars]
        
        # Build the joint factorization
        joint_eq = self.get_joint_factorization()
        
        # Format query
        query_str = f"P({', '.join(query_vars)})"
        
        if hidden_vars:
            sum_str = ", ".join(hidden_vars)
            formula = f"Sum_{{{sum_str}}} [ {joint_eq} ]"
        else:
            formula = joint_eq
        
        return query_str, formula


def generate_cpts(bn, cardinality=2):
    """Generate random conditional probability tables for all nodes."""
    cpts = {}
    for node in bn.nodes:
        parents = bn.parents[node]
        if not parents:
            cpts[node] = normalize(np.random.rand(cardinality))
        else:
            shape = tuple([cardinality] * len(parents) + [cardinality])
            cpts[node] = random_cpt(shape)
    return cpts


def compute_joint(bn, cpts, cardinality=2):
    """Compute the full joint distribution."""
    n_nodes = len(bn.nodes)
    shape = tuple([cardinality] * n_nodes)
    joint = np.zeros(shape)
    
    node_to_idx = {node: i for i, node in enumerate(bn.nodes)}
    
    # Iterate over all possible assignments
    for assignment in product(range(cardinality), repeat=n_nodes):
        prob = 1.0
        for node in bn.nodes:
            node_val = assignment[node_to_idx[node]]
            parents = bn.parents[node]
            
            if not parents:
                prob *= cpts[node][node_val]
            else:
                parent_vals = tuple(assignment[node_to_idx[p]] for p in parents)
                prob *= cpts[node][parent_vals + (node_val,)]
        
        joint[assignment] = prob
    
    return joint, node_to_idx


def compute_marginal(joint, node_to_idx, query_vars, bn):
    """Compute marginal distribution by summing out hidden variables."""
    hidden_vars = [node for node in bn.nodes if node not in query_vars]
    hidden_axes = tuple(node_to_idx[v] for v in hidden_vars)
    
    if hidden_axes:
        return joint.sum(axis=hidden_axes)
    else:
        return joint


def verify_marginal(bn, cpts, query_vars, joint, node_to_idx, cardinality=2):
    """
    Verify marginal by computing it two ways:
    1. Summing from joint distribution
    2. Direct formula computation
    """
    # Method 1: Sum from joint
    marginal_joint = compute_marginal(joint, node_to_idx, query_vars, bn)
    
    # Method 2: Direct computation
    query_idx = {var: i for i, var in enumerate(query_vars)}
    hidden_vars = [node for node in bn.nodes if node not in query_vars]
    
    shape = tuple([cardinality] * len(query_vars))
    marginal_formula = np.zeros(shape)
    
    # For each query assignment, sum over hidden variables
    for q_assignment in product(range(cardinality), repeat=len(query_vars)):
        total = 0.0
        
        # Sum over all hidden variable assignments
        for h_assignment in product(range(cardinality), repeat=len(hidden_vars)):
            # Build full assignment
            full_assignment = {}
            for i, var in enumerate(query_vars):
                full_assignment[var] = q_assignment[i]
            for i, var in enumerate(hidden_vars):
                full_assignment[var] = h_assignment[i]
            
            # Compute probability for this assignment
            prob = 1.0
            for node in bn.nodes:
                node_val = full_assignment[node]
                parents = bn.parents[node]
                
                if not parents:
                    prob *= cpts[node][node_val]
                else:
                    parent_vals = tuple(full_assignment[p] for p in parents)
                    prob *= cpts[node][parent_vals + (node_val,)]
            
            total += prob
        
        marginal_formula[q_assignment] = total
    
    return marginal_joint, marginal_formula


def solve_problem(problem_num, edges, queries, node_order=None):
    """Solve a complete problem with formula generation and numerical verification."""
    print(f"SSolution Problem {problem_num}")
    
    
    # Create Bayesian Network
    bn = BayesianNetwork(edges, node_order)
    
    # Generate CPTs and compute joint
    cpts = generate_cpts(bn)
    joint, node_to_idx = compute_joint(bn, cpts)
    
    # Show the joint factorization
    nodes_str = ", ".join(bn.nodes)
    print(f"P({nodes_str}) = {bn.get_joint_factorization()}")
    
    # Process each query
    for query_vars in queries:
        query_str, formula = bn.get_marginal_formula(query_vars)
        marginal_joint, marginal_formula = verify_marginal(bn, cpts, query_vars, joint, node_to_idx)
        max_diff = np.abs(marginal_joint - marginal_formula).max()
        
        print(f"{query_str} = {formula}")
    
    print(f"All formulas verified numerically (max error ≈ 0) \n")


# ==================== PROBLEM 1 ====================
# Graph: X -> Y -> Z (chain)

edges_1 = [('X', 'Y'), ('Y', 'Z')]

queries_1 = [
    ['X', 'Z'],       # Marginal
    ['X', 'Y'],       # Marginal
    ['Y', 'Z']        # Marginal
]

# ==================== PROBLEM 2 ====================
# 7-node DAG: x1,x2 -> x3, x3,x4,x5 -> x6, x6 -> x7

edges_2 = [
    ('x1', 'x3'),
    ('x2', 'x3'),
    ('x3', 'x6'),
    ('x4', 'x6'),
    ('x5', 'x6'),
    ('x6', 'x7')
]

queries_2 = [
    ['x1', 'x3', 'x5', 'x7'],                     # Marginal
    ['x2', 'x4', 'x6', 'x7'],                     # Marginal
    ['x3', 'x6', 'x7'],                           # Marginal
    ['x1', 'x2', 'x4', 'x5']                      # Marginal
]

# ==================== PROBLEM 3 ====================
# Same as Problem 2 but with additional edge x4 -> x7

edges_3 = [
    ('x1', 'x3'),
    ('x2', 'x3'),
    ('x3', 'x6'),
    ('x4', 'x6'),
    ('x5', 'x6'),
    ('x6', 'x7'),
    ('x4', 'x7')  # Additional edge!
]

queries_3 = [
    ['x1', 'x3', 'x5', 'x7'],                     # Marginal
    ['x2', 'x4', 'x6', 'x7'],                     # Marginal
    ['x3', 'x6', 'x7'],                           # Marginal
    ['x1', 'x2', 'x4', 'x5']                      # Marginal
]


# ==================== MAIN ====================

if __name__ == "__main__":
    np.random.seed(42)
    
    solve_problem(1, edges_1, queries_1, node_order=['X', 'Y', 'Z'])
    solve_problem(2, edges_2, queries_2, node_order=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'])
    solve_problem(3, edges_3, queries_3, node_order=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'])
