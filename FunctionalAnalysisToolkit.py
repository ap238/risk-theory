"""
risk-theory: A library of Python programs demonstrating mathematical algorithms for probability.
Module: FunctionalAnalysisToolkit
This module provides tools for exploring functional analysis concepts, with a focus on their probabilistic and mathematical foundations.
"""
import numpy as np
import matplotlib.pyplot as plt

class FunctionalAnalysisToolkit:
    """
    A toolkit for exploring functional analysis: normed spaces, Banach/Hilbert spaces, operators, duals,
    major theorems, and spectral theory. Includes visualization and demos for many concepts.
    Part of the risk-theory library for demonstrating mathematical algorithms and their probabilistic context.
    """
    # --- Normed Vector Spaces ---
    @staticmethod
    def norm(x, p=2):
        """
        Compute the p-norm of a vector.
        Args:
            x (array-like): Input vector.
            p (int or float): Order of the norm (default 2).
        Returns:
            float: The p-norm of x.
        """
        x = np.array(x)
        return np.linalg.norm(x, ord=p)

    @staticmethod
    def plot_unit_ball(p=2):
        """
        Plot the unit ball in R^2 for a given p-norm.
        Args:
            p (int, float, or 'inf'): Order of the norm.
        """
        theta = np.linspace(0, 2*np.pi, 200)
        if p == 2:
            x = np.cos(theta)
            y = np.sin(theta)
        elif p == 1:
            x = np.sign(np.cos(theta)) * (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))**-1
            y = np.sign(np.sin(theta)) * (np.abs(np.cos(theta)) + np.abs(np.sin(theta)))**-1
        elif p == np.inf:
            x = np.cos(theta) / np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))
            y = np.sin(theta) / np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))
        else:
            x = np.sign(np.cos(theta)) * (np.abs(np.cos(theta))**p + np.abs(np.sin(theta))**p)**(-1/p)
            y = np.sign(np.sin(theta)) * (np.abs(np.cos(theta))**p + np.abs(np.sin(theta))**p)**(-1/p)
        plt.plot(x, y)
        plt.gca().set_aspect('equal')
        plt.title(f'Unit Ball in ℝ², p={p}')
        plt.show()

    # --- Banach Spaces ---
    @staticmethod
    def is_cauchy(seq, tol=1e-6):
        """
        Check if a sequence is Cauchy in the given norm.
        Args:
            seq (list of vectors): Sequence to check.
            tol (float): Tolerance for Cauchy condition.
        Returns:
            bool: True if sequence is Cauchy, False otherwise.
        """
        for i in range(len(seq)):
            for j in range(i+1, len(seq)):
                if np.linalg.norm(np.array(seq[i])-np.array(seq[j])) > tol:
                    return False
        return True

    @staticmethod
    def is_complete(seq, limit, tol=1e-6):
        """
        Check if a sequence converges to a given limit (completeness).
        Args:
            seq (list of vectors): Sequence to check.
            limit (vector): Claimed limit.
            tol (float): Tolerance for convergence.
        Returns:
            bool: True if sequence converges to limit, False otherwise.
        """
        return np.linalg.norm(np.array(seq[-1])-np.array(limit)) < tol

    # --- Hilbert Spaces ---
    @staticmethod
    def inner_product(x, y):
        """
        Compute the inner product of two vectors.
        Args:
            x, y (array-like): Input vectors.
        Returns:
            float: Inner product.
        """
        return np.dot(x, y)

    @staticmethod
    def projection(u, v):
        """
        Project vector u onto vector v.
        Args:
            u, v (array-like): Vectors.
        Returns:
            np.ndarray: Projection of u onto v.
        """
        # Project u onto v
        v = np.array(v)
        u = np.array(u)
        return (np.dot(u, v) / np.dot(v, v)) * v

    @staticmethod
    def plot_orthogonality():
        """
        Visualize orthogonal vectors in R^2.
        """
        plt.arrow(0, 0, 1, 0, head_width=0.05, color='b', label='v')
        plt.arrow(0, 0, 0, 1, head_width=0.05, color='r', label='w')
        plt.xlim(-0.2, 1.2)
        plt.ylim(-0.2, 1.2)
        plt.gca().set_aspect('equal')
        plt.legend(['v', 'w'])
        plt.title('Orthogonal Vectors in ℝ²')
        plt.show()

    # --- Bounded Linear Operators ---
    @staticmethod
    def operator_norm(A):
        """
        Compute the operator norm (2-norm) of a matrix.
        Args:
            A (np.ndarray): Matrix.
        Returns:
            float: Operator norm.
        """
        return np.linalg.norm(A, ord=2)

    # --- Dual Spaces ---
    @staticmethod
    def dual_functional(x, y):
        """
        Compute the value of the linear functional f_y(x) = <x, y>.
        Args:
            x, y (array-like): Vectors.
        Returns:
            float: Value of the functional.
        """
        # Linear functional: f_y(x) = <x, y>
        return np.dot(x, y)

    @staticmethod
    def plot_functional(y):
        """
        Plot the linear functional f_y(x) = <x, y> for varying x.
        Args:
            y (array-like): Vector defining the functional.
        """
        x = np.linspace(-2, 2, 100)
        vals = [np.dot([xi, 1], y) for xi in x]
        plt.plot(x, vals)
        plt.title('Linear Functional f_y(x) = <x, y>')
        plt.xlabel('x')
        plt.ylabel('f_y([x,1])')
        plt.show()

    # --- Hahn-Banach Theorem (demo) ---
    @staticmethod
    def hahn_banach_demo():
        """
        Demonstrate the Hahn-Banach theorem by visualizing extension of a functional.
        """
        # Show extension of a functional from a subspace (visual demo)
        x = np.linspace(-2, 2, 100)
        f = 2*x  # defined on x-axis
        g = 2*x + 1  # extension
        plt.plot(x, f, label='Original functional')
        plt.plot(x, g, label='Extension')
        plt.legend()
        plt.title('Hahn-Banach Extension Demo')
        plt.show()

    # --- Open Mapping Theorem (demo) ---
    @staticmethod
    def open_mapping_demo():
        """
        Demonstrate the open mapping theorem by visualizing the image of an open set.
        """
        # Show image of open set under a surjective bounded operator
        theta = np.linspace(0, 2*np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        A = np.array([[2, 0], [0, 0.5]])
        xy = np.vstack([x, y])
        Ax = A @ xy
        plt.plot(x, y, label='Open Ball')
        plt.plot(Ax[0], Ax[1], label='Image under A')
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.title('Open Mapping Theorem Demo')
        plt.show()

    # --- Uniform Boundedness Principle (demo) ---
    @staticmethod
    def uniform_boundedness_demo():
        """
        Demonstrate the uniform boundedness principle with a family of operators.
        """
        # Show family of operators with bounded sup norm
        x = np.linspace(-1, 1, 100)
        for n in range(1, 6):
            plt.plot(x, n*x, label=f'T_{n}(x)')
        plt.title('Uniform Boundedness Principle Demo')
        plt.xlabel('x')
        plt.ylabel('T_n(x)')
        plt.legend()
        plt.show()

    # --- Spectral Theory (compact, self-adjoint) ---
    @staticmethod
    def spectral_theory_demo():
        """
        Demonstrate spectral theory for a compact, self-adjoint operator (symmetric matrix).
        """
        # Compact, self-adjoint operator: symmetric matrix
        A = np.array([[2, 1], [1, 2]])
        eigvals, eigvecs = np.linalg.eig(A)
        print('Eigenvalues:', eigvals)
        print('Eigenvectors (columns):\n', eigvecs)
        plt.scatter(eigvals.real, eigvals.imag, c='b', marker='o')
        plt.axhline(0, color='gray', lw=0.5)
        plt.axvline(0, color='gray', lw=0.5)
        plt.title('Spectrum of Compact, Self-Adjoint Operator')
        plt.xlabel('Real part')
        plt.ylabel('Imaginary part')
        plt.grid(True)
        plt.show()

# Example usage:
if __name__ == "__main__":
    print(__doc__)
    # Normed vector spaces
    print('Norm of [3,4]:', FunctionalAnalysisToolkit.norm([3,4]))
    FunctionalAnalysisToolkit.plot_unit_ball(p=2)
    # Banach spaces
    seq = [[1/n, 0] for n in range(1, 20)]
    print('Is Cauchy:', FunctionalAnalysisToolkit.is_cauchy(seq))
    print('Is Complete:', FunctionalAnalysisToolkit.is_complete(seq, [0,0]))
    # Hilbert spaces
    print('Inner product:', FunctionalAnalysisToolkit.inner_product([1,2],[3,4]))
    print('Projection of [2,2] onto [1,0]:', FunctionalAnalysisToolkit.projection([2,2],[1,0]))
    FunctionalAnalysisToolkit.plot_orthogonality()
    # Bounded linear operators
    A = np.array([[1,2],[3,4]])
    print('Operator norm:', FunctionalAnalysisToolkit.operator_norm(A))
    # Dual spaces
    print('Dual functional:', FunctionalAnalysisToolkit.dual_functional([1,2],[3,4]))
    FunctionalAnalysisToolkit.plot_functional([2,1])
    # Hahn-Banach
    FunctionalAnalysisToolkit.hahn_banach_demo()
    # Open mapping theorem
    FunctionalAnalysisToolkit.open_mapping_demo()
    # Uniform boundedness
    FunctionalAnalysisToolkit.uniform_boundedness_demo()
    # Spectral theory
    FunctionalAnalysisToolkit.spectral_theory_demo() 