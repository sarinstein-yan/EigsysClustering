import numpy as np
import sympy as sp
from .util import kron_batched

def shift_matrix(N, shift=1, pbc=True):
    """
    Constructs the translation (shift) matrix for a 1D chain of length N.

    For a given shift value, the function creates a matrix that translates (or "shifts") 
    the basis elements. For periodic boundary conditions (pbc=True), the matrix is rolled; 
    otherwise, a diagonal matrix with shifted ones is returned.

    Parameters:
        N (int): The number of sites in the 1D chain.
        shift (int, optional): The number of sites to shift (default is 1).
        pbc (bool, optional): Whether to enforce periodic boundary conditions (default is True).

    Returns:
        A NumPy array representing the shift matrix. For N <= 1, returns a 1x1 identity matrix.
    """
    if N is None or N <= 1:
        return np.eye(1)
    if pbc:
        return np.roll(np.eye(N), shift, axis=-1)
    else:
        elem = np.ones(N - abs(shift))
        return np.diag(elem, k=shift)

# --- for 1D Bloch Hamiltonian --- #
def hk2hz_1d(h_k, k, z):
    """
    Converts a 1D Bloch Hamiltonian from momentum-space (k) to a representation 
    in terms of a complex variable (z) via logarithmic substitution k = -i log(z)

    Parameters:
        h_k: A sympy Matrix or expression representing the Bloch Hamiltonian.
        k: sympy symbol corresponding to the momentum.
        z (optional): Complex variable for substitution (e.g. z = exp(i*k)).

    Returns:
        A simplified sympy expression or Matrix for the Hamiltonian expressed in 
        terms of z.
    """
    subs_dict = {k: sp.log(z)/sp.I}
    if isinstance(h_k, sp.Matrix):
        H_sub = h_k.applyfunc(lambda expr: expr.subs(subs_dict))
        h_z = H_sub.applyfunc(lambda expr: sp.simplify(expr.rewrite(sp.exp)))
    else:
        H_sub = h_k.subs(subs_dict)
        h_z = sp.simplify(H_sub.rewrite(sp.exp))
    return h_z

def expand_hz_as_hop_dict_1d(h_z, z):
    """
    Expands a 1D Hamiltonian expressed in terms of the complex variable (z) into its polynomial form.

    The function parses each matrix element of h_z and extracts the coefficients corresponding to 
    different powers of z. It returns a dictionary mapping the exponent (an integer) to the 
    corresponding coefficient matrix.

    Parameters:
        h_z: A sympy Matrix or expression representing the Hamiltonian in terms of z.
        z: sympy symbol corresponding to the complex variable (typically exp(i*k)).

    Returns:
        A dictionary where:
            - Keys are integers representing the exponent of z.
            - Values are sympy Matrices with the corresponding coefficients.
    """
    if not isinstance(h_z, sp.Matrix):
        h_z = sp.Matrix(h_z)

    d_rows, d_cols = h_z.shape
    poly_dict = {}

    for i in range(d_rows):
        for j in range(d_cols):
            expr = sp.expand(h_z[i, j])
            terms = expr.as_ordered_terms()
            for term in terms:
                coeff, factors = term.as_coeff_mul()
                prod = sp.Mul(*factors)
                pdict = prod.as_powers_dict()
                exp_z = pdict.get(z, 0)
                remainder = sp.simplify(prod / (z**exp_z))
                term_coeff = coeff * remainder
                key = int(exp_z)
                if key not in poly_dict:
                    poly_dict[key] = sp.zeros(d_rows, d_cols)
                poly_dict[key][i, j] += term_coeff
    return poly_dict

def H_1D_batch(h_k, k, N, param_dict):
    """
    Constructs the 1D Hamiltonian matrices for both open (OBC) and periodic (PBC) boundary conditions 
    from a given Bloch Hamiltonian for a 1D lattice system.

    Parameters
    ----------
    h_k : sympy expression or sympy Matrix
        The Bloch Hamiltonian expressed in momentum space as a function of the momentum symbol k.
    k : sympy symbol
        The momentum symbol used in the Bloch Hamiltonian.
    N : int
        The number of lattice sites in the 1D system.
    param_dict : dict
        A dictionary mapping sympy symbols to numerical values (or arrays) for batch processing. 
        All parameter arrays must share the same shape.
    
    Returns
    -------
    Hobc_arr : np.ndarray
        The Hamiltonian matrix under open boundary conditions with shape (*batch_shape, N*d, N*d),
        where d is the dimension of the internal degree of freedom.
    Hpbc_arr : np.ndarray
        The Hamiltonian matrix under periodic boundary conditions with the same shape as Hobc_arr.
    """
    # Check that all parameter arrays have the same shape
    batch_shapes = [np.shape(v) for v in param_dict.values()]
    assert all(shape == batch_shapes[0] for shape in batch_shapes), "Parameter values must have the same shape."
    batch_shape = batch_shapes[0]
    # Define dummy symbols for the phase factor z and momentum k.
    z = sp.Symbol('z')
    h_z = hk2hz_1d(h_k, k, z)
    hoppings = expand_hz_as_hop_dict_1d(h_z, z)

    Hobc_arr = None
    Hpbc_arr = None

    for shift, val in hoppings.items():
        Tobc = shift_matrix(N, shift, pbc=False).astype(np.complex128)
        Tpbc = shift_matrix(N, shift, pbc=True).astype(np.complex128)
        Tobc_arr = np.tile(Tobc, (*batch_shape, 1, 1))
        Tpbc_arr = np.tile(Tpbc, (*batch_shape, 1, 1))

        if len(val.free_symbols) == 0:
            hop = np.array(val.tolist(), dtype=np.complex128)
            hop_arr = np.tile(hop, (*batch_shape, 1, 1))
        else:
            hop_arr = np.zeros((*batch_shape, *val.shape), dtype=np.complex128)
            # Sort free symbols to ensure a consistent order with param_dict keys.
            f_vars = sorted(val.free_symbols, key=lambda s: s.name)
            for idx in np.ndindex(val.shape):
                f = sp.lambdify(f_vars, val[idx], 'numpy')
                # Create the argument list for the free symbols based on the sorted order.
                args = [param_dict[s] for s in f_vars]
                hop_arr[..., *idx] = f(*args)

        Hobc_temp = kron_batched(Tobc_arr, hop_arr)
        Hpbc_temp = kron_batched(Tpbc_arr, hop_arr)

        if Hobc_arr is None or Hpbc_arr is None:
            Hobc_arr = Hobc_temp
            Hpbc_arr = Hpbc_temp
        else:
            Hobc_arr += Hobc_temp
            Hpbc_arr += Hpbc_temp

    return Hobc_arr, Hpbc_arr






# --- for general 1D-3D Bloch Hamiltonian --- #
def hk2hz(h_k, kx, ky, kz, zx=None, zy=None, zz=None):
    """
    Converts a Bloch Hamiltonian from momentum-space (kx, ky, kz) to a representation 
    in terms of complex variables (zx, zy, zz) via logarithmic substitution.

    For each momentum variable, if the corresponding complex variable is provided and 
    the momentum symbol appears in h_k, the substitution k = -i log(z) is performed.
    Otherwise, no substitution is done for that symbol.

    Parameters:
        h_k: A sympy Matrix or expression representing the Bloch Hamiltonian.
        kx, ky, kz: sympy symbols corresponding to the momentum components.
        zx, zy, zz (optional): Complex variables for substitution (e.g. zx = exp(i*kx)).
            If any is None, or the corresponding momentum symbol is not present in h_k,
            that variable remains unchanged.

    Returns:
        A simplified sympy expression or Matrix for the Hamiltonian expressed in terms of zx, zy, zz.
    """
    # Collect all free symbols from h_k.
    if isinstance(h_k, sp.Matrix):
        free_syms = set().union(*[expr.free_symbols for expr in h_k])
    else:
        free_syms = h_k.free_symbols

    # Only add the substitution if k is in free_syms and the corresponding z is provided.
    subs_dict = {}
    for k, z in [(kx, zx), (ky, zy), (kz, zz)]:
        if k in free_syms and z is not None:
            subs_dict[k] = sp.log(z) / sp.I

    # Apply substitution.
    if isinstance(h_k, sp.Matrix):
        H_sub = h_k.applyfunc(lambda expr: expr.subs(subs_dict))
    else:
        H_sub = h_k.subs(subs_dict)

    return sp.simplify(H_sub.rewrite(sp.exp))

def expand_hz_as_hop_dict(h_z, zx=None, zy=None, zz=None):
    """
    Expands a Hamiltonian expressed in terms of complex variables (zx, zy, zz) into its polynomial form.

    The function parses each matrix element of h_z and extracts the coefficients corresponding to
    different powers of the provided complex variables. It returns a dictionary mapping the power tuple
    (for the provided z variables, in order) to the corresponding coefficient matrix.

    Parameters:
        h_z: A sympy Matrix or expression representing the Hamiltonian in terms of zx, zy, zz.
        zx, zy, zz (optional): sympy symbols corresponding to the complex variables (typically exp(i*kx), etc.).
            If any is None, that variable is ignored in the expansion.

    Returns:
        A dictionary where:
            - Keys are tuples of integers representing the exponents of the provided z variables.
            - Values are sympy Matrices with the corresponding coefficients.
    """
    if not isinstance(h_z, sp.Matrix):
        h_z = sp.Matrix(h_z)

    d_rows, d_cols = h_z.shape
    poly_dict = {}

    # Build list of provided z variables in order.
    z_vars = [z for z in (zx, zy, zz) if z is not None]

    for i in range(d_rows):
        for j in range(d_cols):
            expr = sp.expand(h_z[i, j])
            for term in expr.as_ordered_terms():
                coeff, factors = term.as_coeff_mul()
                prod = sp.Mul(*factors)
                pdict = prod.as_powers_dict()
                exponents = []
                divisor = 1
                for z in z_vars:
                    exp = pdict.get(z, 0)
                    exponents.append(int(exp))
                    divisor *= z**exp
                remainder = sp.simplify(prod / divisor)
                term_coeff = coeff * remainder
                key = tuple(exponents)
                if key not in poly_dict:
                    poly_dict[key] = sp.zeros(d_rows, d_cols)
                poly_dict[key][i, j] += term_coeff
    return poly_dict