import numpy as np
import matplotlib.pyplot as plt
import itertools
import networkx as nx

class TopologyToolkit:
    """
    A toolkit for exploring topology: open/closed sets, basis, continuity, compactness, connectedness,
    metric spaces, fundamental group, homotopy, homology, and covering spaces.
    Includes visualization for many concepts.
    """
    # --- Open and Closed Sets ---
    @staticmethod
    def plot_open_ball(center, radius):
        fig, ax = plt.subplots()
        circle = plt.Circle(center, radius, color='blue', fill=False, linestyle='--', label='Open Ball')
        ax.add_artist(circle)
        ax.set_xlim(center[0]-radius-1, center[0]+radius+1)
        ax.set_ylim(center[1]-radius-1, center[1]+radius+1)
        ax.set_aspect('equal')
        plt.scatter(*center, color='red', label='Center')
        plt.legend()
        plt.title('Open Ball in ℝ²')
        plt.show()

    @staticmethod
    def is_open_set(points, open_balls, epsilon=1e-6):
        # points: set of points in R^2, open_balls: list of (center, radius)
        for p in points:
            if not any(np.linalg.norm(np.array(p)-np.array(c)) < r-epsilon for c, r in open_balls):
                return False
        return True

    # --- Basis and Subbasis ---
    @staticmethod
    def basis_from_balls(centers, radii):
        return [(c, r) for c in centers for r in radii]

    # --- Continuous Maps ---
    @staticmethod
    def plot_continuous_map(f, domain, num=100):
        x = np.linspace(domain[0], domain[1], num)
        y = f(x)
        plt.plot(x, y)
        plt.title('Continuous Map')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.show()

    # --- Compactness (finite/visualizable spaces) ---
    @staticmethod
    def is_compact(points, open_covers):
        # For finite sets: compact iff every open cover has a finite subcover
        for cover in open_covers:
            covered = set()
            for c, r in cover:
                covered |= {p for p in points if np.linalg.norm(np.array(p)-np.array(c)) < r}
            if covered == set(points):
                return True
        return False

    # --- Connectedness ---
    @staticmethod
    def is_connected(points, edges):
        # points: list of points, edges: list of (i,j) pairs
        G = nx.Graph()
        G.add_nodes_from(range(len(points)))
        G.add_edges_from(edges)
        return nx.is_connected(G)

    # --- Metric Spaces ---
    @staticmethod
    def distance(p, q):
        return np.linalg.norm(np.array(p)-np.array(q))

    @staticmethod
    def plot_metric_space(points):
        x, y = zip(*points)
        plt.scatter(x, y)
        plt.title('Metric Space Points')
        plt.show()

    # --- Fundamental Group (simple spaces) ---
    @staticmethod
    def fundamental_group_circle(num_loops=3):
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), label='S¹')
        for k in range(1, num_loops+1):
            plt.plot(np.cos(k*theta), np.sin(k*theta), linestyle='--', label=f'Loop {k}')
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.title('Loops in S¹ (Fundamental Group)')
        plt.show()

    # --- Homotopy (visual/animation stub) ---
    @staticmethod
    def plot_homotopy():
        # Animate a homotopy between two loops (stub: just show start/end)
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), label='Loop 1')
        plt.plot(0.5*np.cos(theta), 0.5*np.sin(theta), label='Loop 2')
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.title('Homotopy between Loops (Start/End)')
        plt.show()

    # --- Homology Groups (simplicial, singular, for simple complexes) ---
    @staticmethod
    def plot_simplicial_complex():
        # Simple triangle (1-simplex, 2-simplex)
        points = np.array([[0,0],[1,0],[0.5,np.sqrt(3)/2]])
        plt.plot(*zip(*(points.tolist()+[points[0].tolist()])), marker='o', label='2-simplex')
        plt.fill(*zip(*points), alpha=0.2)
        plt.scatter(points[:,0], points[:,1], color='red')
        plt.title('Simplicial Complex (Triangle)')
        plt.legend()
        plt.show()

    # --- Covering Spaces ---
    @staticmethod
    def plot_covering_space():
        # Visualize covering of S¹ by R (universal cover)
        theta = np.linspace(0, 4*np.pi, 200)
        plt.plot(np.cos(theta), np.sin(theta), label='S¹')
        plt.plot(theta/(2*np.pi), np.zeros_like(theta), label='R (cover)')
        for k in range(5):
            plt.plot([k, k], [0, 1], color='gray', linestyle='--', alpha=0.5)
        plt.legend()
        plt.title('Covering Space: R → S¹')
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Open/closed sets
    TopologyToolkit.plot_open_ball((0,0), 2)
    print("Is open set:", TopologyToolkit.is_open_set([(0,0)], [((0,0),2)]))
    # Basis
    print("Basis from balls:", TopologyToolkit.basis_from_balls([(0,0),(1,1)], [1,2]))
    # Continuous map
    TopologyToolkit.plot_continuous_map(np.sin, (0, 2*np.pi))
    # Compactness
    print("Is compact:", TopologyToolkit.is_compact([(0,0),(1,0)], [[((0,0),1),((1,0),1)]]))
    # Connectedness
    print("Is connected:", TopologyToolkit.is_connected([(0,0),(1,0)], [(0,1)]))
    # Metric space
    print("Distance:", TopologyToolkit.distance((0,0),(1,1)))
    TopologyToolkit.plot_metric_space([(0,0),(1,0),(0,1)])
    # Fundamental group
    TopologyToolkit.fundamental_group_circle(num_loops=2)
    # Homotopy
    TopologyToolkit.plot_homotopy()
    # Homology
    TopologyToolkit.plot_simplicial_complex()
    # Covering space
    TopologyToolkit.plot_covering_space() 