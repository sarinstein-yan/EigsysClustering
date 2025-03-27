import numpy as np
import math
from itertools import combinations
from scipy import sparse as sp
import tensorflow as tf
import multiprocessing as mp


def de2bi_modified(d, n):
    """
    Convert a decimal number d into its binary representation with n bits.
    Most-significant bit first.
    (MATLAB: b = rem(floor(d*pow2(1-n:0)),2);)
    """
    # Using bit shifting for integer d:
    b = np.array([(d >> (n - 1 - i)) & 1 for i in range(n)], dtype=int)
    return b

def bi2de_modified(b2):
    """
    Convert a binary representation (array-like, MSB first) to a decimal integer.
    (MATLAB: d = fliplr(b2)*[2.^((0:n-1))]';)
    """
    d = 0
    n = len(b2)
    # Process bits from least-significant (rightmost) to most-significant.
    for i, bit in enumerate(reversed(b2)):
        d += bit * (2 ** i)
    return d

def binaraysearchasc(x, sval):
    """
    Perform binary search in an ascending sorted list x for the value sval.
    Returns the index if found, otherwise returns None.
    
    (MATLAB equivalent uses 1-based indices; here indices start at 0.)
    """
    left = 0
    right = len(x) - 1
    while left <= right:
        mid = (left + right) // 2  # floor division
        diff = x[mid] - sval
        if diff == 0:
            return mid
        elif diff < 0:
            left = mid + 1
        else:
            right = mid - 1
    return None

def generateBasis_soc(NUM_SITES, NUM_PAR):
    """
    Generate the basis for a system with NUM_SITES and NUM_PAR particles.
    For each valid spin-up sector (with NUM_UP particles, where NUM_DN = NUM_PAR - NUM_UP),
    the function computes all Fock states (represented as integers) for spin-up and spin-down,
    and then forms the combined basis as the Cartesian product.
    
    Returns:
      combinedBasis: an array of shape (TOTAL_ALL_STATES, 3) where the first column 
                     contains the 0-based index, the second column the spin-down state, 
                     and the third the spin-up state.
      TOTAL_ALL_STATES: total number of states (combination of 2*NUM_SITES choose NUM_PAR)
      upStates: concatenated spin-up states (as integers) over the sectors
      dnStates: concatenated spin-down states (as integers) over the sectors
      nosOfState_up, nosOfState_dn, nosOfState_all: cumulative counts for each sector.
      
    (Note: MATLAB code uses 1-based indexing; here all indices and offsets are 0-based.)
    """
    # Total number of states satisfying particle conservation:
    TOTAL_ALL_STATES = math.comb(2 * NUM_SITES, NUM_PAR)
    combinedBasis = np.zeros((TOTAL_ALL_STATES, 3), dtype=int)
    combinedBasis_mid = np.zeros((TOTAL_ALL_STATES, 2), dtype=int)
    
    # Preallocate cumulative arrays (length = NUM_PAR + 2, as in MATLAB)
    nosOfState_up = np.zeros(NUM_PAR + 2, dtype=int)
    nosOfState_dn = np.zeros(NUM_PAR + 2, dtype=int)
    nosOfState_all = np.zeros(NUM_PAR + 2, dtype=int)
    
    # Lists to accumulate the spin states
    upStates_list = []
    dnStates_list = []
    
    # Offsets for concatenation (these play the role of the cumulative counters)
    offset_up = 0
    offset_dn = 0
    offset_all = 0
    
    # Determine the valid range for the number of spin-up electrons:
    L = max(0, NUM_PAR - NUM_SITES)
    U = min(NUM_PAR, NUM_SITES)
    
    def generate_states(num_sites, num_ones):
        """
        Generate a sorted list of integers whose binary representation (with num_sites bits)
        has exactly num_ones ones.
        """
        states = []
        for comb in combinations(range(num_sites), num_ones):
            # Each combination gives positions (0-indexed, with 0 = leftmost bit)
            state = 0
            for pos in comb:
                # Set bit at position corresponding to a weight of 2^(num_sites-1-pos)
                state += 2 ** (num_sites - 1 - pos)
            states.append(state)
        return sorted(states)
    
    # Loop over valid spin-up particle numbers.
    for id_part_up in range(L, U + 1):
        NUM_UP = id_part_up
        NUM_DN = NUM_PAR - NUM_UP
        
        # For spin-up, number of states is C(NUM_SITES, NUM_UP)
        TOTAL_UP_STATES_SF = math.comb(NUM_SITES, NUM_UP)
        # Update cumulative counter: (MATLAB: nosOfState_up(id_part_up+2)= nosOfState_up(id_part_up+1)+TOTAL_UP_STATES_SF)
        nosOfState_up[id_part_up + 1] = nosOfState_up[id_part_up] + TOTAL_UP_STATES_SF
        # Generate all up-states with NUM_UP ones.
        upStates_SF = generate_states(NUM_SITES, NUM_UP)
        upStates_list.extend(upStates_SF)
        offset_up += TOTAL_UP_STATES_SF
        
        # For spin-down, number of states is C(NUM_SITES, NUM_DN)
        TOTAL_DN_STATES_SF = math.comb(NUM_SITES, NUM_DN)
        nosOfState_dn[id_part_up + 1] = nosOfState_dn[id_part_up] + TOTAL_DN_STATES_SF
        dnStates_SF = generate_states(NUM_SITES, NUM_DN)
        dnStates_list.extend(dnStates_SF)
        offset_dn += TOTAL_DN_STATES_SF
        
        # Total states for the current sector (spin-up and spin-down are independent)
        TOTAL_ALL_STATES_SF = TOTAL_UP_STATES_SF * TOTAL_DN_STATES_SF
        nosOfState_all[id_part_up + 1] = nosOfState_all[id_part_up] + TOTAL_ALL_STATES_SF
        
        # Create the combined basis for this sector as the Cartesian product.
        # The ordering is such that for each up-state, we loop over all down-states.
        sector_basis = np.zeros((TOTAL_ALL_STATES_SF, 2), dtype=int)
        k = 0
        for u in upStates_SF:
            for d in dnStates_SF:
                # Each row: [dn_state, up_state]
                sector_basis[k, :] = [d, u]
                k += 1
        
        # Place the sector's combined basis into the overall array.
        combinedBasis_mid[offset_all: offset_all + TOTAL_ALL_STATES_SF, :] = sector_basis
        offset_all += TOTAL_ALL_STATES_SF
    
    # Finalize combinedBasis:
    # First column is the state index (0-based), then spin-down and spin-up states.
    combinedBasis[:, 0] = np.arange(TOTAL_ALL_STATES)
    combinedBasis[:, 1:3] = combinedBasis_mid
    
    # Convert the accumulated state lists to numpy arrays.
    upStates = np.array(upStates_list, dtype=int)
    dnStates = np.array(dnStates_list, dtype=int)
    
    return combinedBasis, TOTAL_ALL_STATES, upStates, dnStates, nosOfState_up, nosOfState_dn, nosOfState_all

def H_set_hubbard_soc(t_s, alpha, t_so, mz, U, noOfSites, noOfPar):
    """
    Generate the Hubbard Hamiltonian with spin-orbit coupling.
    
    Parameters:
      t_s, alpha, t_so, mz, U: model parameters.
      noOfSites: number of lattice sites.
      noOfPar: total number of particles.
      
    Returns:
      totalHamiltonian, kineticHamiltonian, potentialHamiltonian,
      mzHamiltonian, SOCHamiltonian  as sparse matrices.
    
    Note: The helper functions de2bi_modified, bi2de_modified,
          binaraysearchasc (returns 0-based index) and generateBasis_SOC
          are assumed to be defined.
    """
    # Generate basis (assumed to be provided in proper Python/0-index format)
    (combinedBasis, totalNoOfPossiblestates, upStates, dnStates,
     nosOfState_up, nosOfState_dn, nosOfState_all) = generateBasis_soc(noOfSites, noOfPar)
    
    # ---- Potential and Zeeman terms ----
    potential_elems = np.zeros(totalNoOfPossiblestates)
    mz_elems = np.zeros(totalNoOfPossiblestates)
    # In MATLAB, combinedBasis(:,2) and combinedBasis(:,3) were used;
    # here we assume that column 1 (index 1) corresponds to "up" and column 2 (index 2) to "dn" for the potential term.
    extracted_up_states = combinedBasis[:, 1]
    extracted_dn_states = combinedBasis[:, 2]
    
    for j in range(totalNoOfPossiblestates):
        upSectorDec = extracted_up_states[j]
        dnSectorDec = extracted_dn_states[j]
        upSector = de2bi_modified(upSectorDec, noOfSites)
        dnSector = de2bi_modified(dnSectorDec, noOfSites)
        # Bitwise AND to get double occupancy:
        doubleOccupancy = np.bitwise_and(upSector, dnSector)
        potential_elems[j] = np.sum(doubleOccupancy) * U
        mz_elems[j] = mz * (np.sum(upSector) - np.sum(dnSector))
    
    potentialHamiltonian = sp.diags(potential_elems, offsets=0, shape=(totalNoOfPossiblestates, totalNoOfPossiblestates), format='csr')
    mzHamiltonian = sp.diags(mz_elems, offsets=0, shape=(totalNoOfPossiblestates, totalNoOfPossiblestates), format='csr')
    
    # ---- Kinetic term ----
    kinetic_rows = []
    kinetic_cols = []
    kinetic_elems = []
    
    # Loop over all basis states (MATLAB: for m=1:totalNoOfPossiblestates)
    for m in range(totalNoOfPossiblestates):
        # For the kinetic term the ordering is swapped:
        # MATLAB: upSectorDec = combinedBasis(m,3) and dnSectorDec = combinedBasis(m,2)
        upSectorDec = combinedBasis[m, 2]
        dnSectorDec = combinedBasis[m, 1]
        upSector = de2bi_modified(upSectorDec, noOfSites)
        dnSector = de2bi_modified(dnSectorDec, noOfSites)
        noOfup = int(np.sum(upSector))
        noOfUpInterior = noOfup - 1
        noOfDnInterior = noOfPar - noOfup - 1
        
        # Find occupied site indices (0-indexed)
        upNonZero = np.nonzero(upSector)[0]
        dnNonZero = np.nonzero(dnSector)[0]
        
        # --- Spin up shifts ---
        for n in upNonZero:
            # Left shift: (MATLAB: mod(n-2,noOfSites)+1)
            leftShiftedIndex = (n - 1) % noOfSites
            if upSector[leftShiftedIndex] != 1:
                leftShiftResult = upSector.copy()
                leftShiftResult[n] = 0
                leftShiftResult[leftShiftedIndex] = 1
                # Find index in the up-sector basis:
                # MATLAB used: upStates(nosOfState_up(noOfup+1)+1 : nosOfState_up(noOfup+2))
                # In Python (with arrays pre-adjusted), we use:
                up_slice = upStates[ nosOfState_up[noOfup] : nosOfState_up[noOfup+1] ]
                upIndexOfLeftShiftedResult = binaraysearchasc(up_slice, bi2de_modified(leftShiftResult))
                # Determine corresponding dn index:
                dnIndexOfLeftShiftedResult = ((m - nosOfState_all[noOfup]) % (nosOfState_dn[noOfup+1] - nosOfState_dn[noOfup]))
                basisIndexOfLeftShiftedResult = ( nosOfState_all[noOfup] +
                    upIndexOfLeftShiftedResult * (nosOfState_dn[noOfup+1] - nosOfState_dn[noOfup]) +
                    dnIndexOfLeftShiftedResult )
                
                if leftShiftedIndex < n:
                    kinetic_rows.append(basisIndexOfLeftShiftedResult)
                    kinetic_cols.append(m)
                    kinetic_elems.append( -(t_s - alpha) )
                else:
                    if noOfUpInterior % 2 == 0:
                        kinetic_rows.append(basisIndexOfLeftShiftedResult)
                        kinetic_cols.append(m)
                        kinetic_elems.append( -(t_s - alpha) )
                    else:
                        kinetic_rows.append(basisIndexOfLeftShiftedResult)
                        kinetic_cols.append(m)
                        kinetic_elems.append( t_s - alpha )
            
            # Right shift: (MATLAB: mod(n, noOfSites)+1)
            rightShiftedIndex = (n + 1) % noOfSites
            if upSector[rightShiftedIndex] != 1:
                rightShiftResult = upSector.copy()
                rightShiftResult[n] = 0
                rightShiftResult[rightShiftedIndex] = 1
                up_slice = upStates[ nosOfState_up[noOfup] : nosOfState_up[noOfup+1] ]
                upIndexOfRightShiftedResult = binaraysearchasc(up_slice, bi2de_modified(rightShiftResult))
                dnIndexOfRightShiftedResult = ((m - nosOfState_all[noOfup]) % (nosOfState_dn[noOfup+1] - nosOfState_dn[noOfup]))
                basisIndexOfRightShiftedResult = ( nosOfState_all[noOfup] +
                    upIndexOfRightShiftedResult * (nosOfState_dn[noOfup+1] - nosOfState_dn[noOfup]) +
                    dnIndexOfRightShiftedResult )
                
                if rightShiftedIndex > n:
                    kinetic_rows.append(basisIndexOfRightShiftedResult)
                    kinetic_cols.append(m)
                    kinetic_elems.append( -(t_s + alpha) )
                else:
                    if noOfUpInterior % 2 == 0:
                        kinetic_rows.append(basisIndexOfRightShiftedResult)
                        kinetic_cols.append(m)
                        kinetic_elems.append( -(t_s + alpha) )
                    else:
                        kinetic_rows.append(basisIndexOfRightShiftedResult)
                        kinetic_cols.append(m)
                        kinetic_elems.append( t_s + alpha )
        
        # --- Spin down shifts ---
        for p in dnNonZero:
            # Left shift for spin down:
            leftShiftedIndex = (p - 1) % noOfSites
            if dnSector[leftShiftedIndex] != 1:
                leftShiftResult = dnSector.copy()
                leftShiftResult[p] = 0
                leftShiftResult[leftShiftedIndex] = 1
                dn_slice = dnStates[ nosOfState_dn[noOfup] : nosOfState_dn[noOfup+1] ]
                dnIndexOfLeftShiftedResult = binaraysearchasc(dn_slice, bi2de_modified(leftShiftResult))
                upIndexOfLeftShiftedResult = (m - nosOfState_all[noOfup]) // (nosOfState_dn[noOfup+1] - nosOfState_dn[noOfup])
                basisIndexOfLeftShiftedResult = ( nosOfState_all[noOfup] +
                    upIndexOfLeftShiftedResult * (nosOfState_dn[noOfup+1] - nosOfState_dn[noOfup]) +
                    dnIndexOfLeftShiftedResult )
                
                if leftShiftedIndex < p:
                    kinetic_rows.append(basisIndexOfLeftShiftedResult)
                    kinetic_cols.append(m)
                    kinetic_elems.append( t_s - alpha )
                else:
                    if noOfDnInterior % 2 == 0:
                        kinetic_rows.append(basisIndexOfLeftShiftedResult)
                        kinetic_cols.append(m)
                        kinetic_elems.append( t_s - alpha )
                    else:
                        kinetic_rows.append(basisIndexOfLeftShiftedResult)
                        kinetic_cols.append(m)
                        kinetic_elems.append( -(t_s - alpha) )
            
            # Right shift for spin down:
            rightShiftedIndex = (p + 1) % noOfSites
            if dnSector[rightShiftedIndex] != 1:
                rightShiftResult = dnSector.copy()
                rightShiftResult[p] = 0
                rightShiftResult[rightShiftedIndex] = 1
                dn_slice = dnStates[ nosOfState_dn[noOfup] : nosOfState_dn[noOfup+1] ]
                dnIndexOfRightShiftedResult = binaraysearchasc(dn_slice, bi2de_modified(rightShiftResult))
                upIndexOfRightShiftedResult = (m - nosOfState_all[noOfup]) // (nosOfState_dn[noOfup+1] - nosOfState_dn[noOfup])
                basisIndexOfRightShiftedResult = ( nosOfState_all[noOfup] +
                    upIndexOfRightShiftedResult * (nosOfState_dn[noOfup+1] - nosOfState_dn[noOfup]) +
                    dnIndexOfRightShiftedResult )
                
                if rightShiftedIndex > p:
                    kinetic_rows.append(basisIndexOfRightShiftedResult)
                    kinetic_cols.append(m)
                    kinetic_elems.append( t_s + alpha )
                else:
                    if noOfDnInterior % 2 == 0:
                        kinetic_rows.append(basisIndexOfRightShiftedResult)
                        kinetic_cols.append(m)
                        kinetic_elems.append( t_s + alpha )
                    else:
                        kinetic_rows.append(basisIndexOfRightShiftedResult)
                        kinetic_cols.append(m)
                        kinetic_elems.append( -(t_s + alpha) )
    
    kinetic_rows = np.array(kinetic_rows)
    kinetic_cols = np.array(kinetic_cols)
    kinetic_elems = np.array(kinetic_elems)
    kineticHamiltonian = sp.csr_matrix((kinetic_elems, (kinetic_rows, kinetic_cols)),
                                         shape=(totalNoOfPossiblestates, totalNoOfPossiblestates))
    
    # ---- SOC term ----
    SOC_rows = []
    SOC_cols = []
    SOC_elems = []
    
    for m in range(totalNoOfPossiblestates):
        # For SOC, again swap the order:
        upSectorDec = combinedBasis[m, 2]
        dnSectorDec = combinedBasis[m, 1]
        upSector = de2bi_modified(upSectorDec, noOfSites)
        dnSector = de2bi_modified(dnSectorDec, noOfSites)
        noOfup = int(np.sum(upSector))
        upNonZero = np.nonzero(upSector)[0]
        dnNonZero = np.nonzero(dnSector)[0]
        
        # --- Flipping from spin up to spin down ---
        for n in upNonZero:
            leftShiftedIndex = (n - 1) % noOfSites
            if dnSector[leftShiftedIndex] != 1:
                leftShiftResult_up = upSector.copy()
                leftShiftResult_dn = dnSector.copy()
                leftShiftResult_up[n] = 0
                leftShiftResult_dn[leftShiftedIndex] = 1
                # After flip, the up count decreases by one.
                up_slice = upStates[ nosOfState_up[noOfup-1] : nosOfState_up[noOfup] ]
                upIndexOfLeftShiftedResult = binaraysearchasc(up_slice, bi2de_modified(leftShiftResult_up))
                dn_slice = dnStates[ nosOfState_dn[noOfup-1] : nosOfState_dn[noOfup] ]
                dnIndexOfLeftShiftedResult = binaraysearchasc(dn_slice, bi2de_modified(leftShiftResult_dn))
                basisIndexOfLeftShiftedResult = ( nosOfState_all[noOfup-1] +
                    upIndexOfLeftShiftedResult * (nosOfState_dn[noOfup] - nosOfState_dn[noOfup-1]) +
                    dnIndexOfLeftShiftedResult )
                phase = (1 - 2 * ((np.sum(upSector[:n+1]) + np.sum(dnSector[:leftShiftedIndex]) + noOfup - 2) % 2))
                SOC_rows.append(basisIndexOfLeftShiftedResult)
                SOC_cols.append(m)
                SOC_elems.append(phase * t_so)
            
            rightShiftedIndex = (n + 1) % noOfSites
            if dnSector[rightShiftedIndex] != 1:
                rightShiftResult_up = upSector.copy()
                rightShiftResult_dn = dnSector.copy()
                rightShiftResult_up[n] = 0
                rightShiftResult_dn[rightShiftedIndex] = 1
                up_slice = upStates[ nosOfState_up[noOfup-1] : nosOfState_up[noOfup] ]
                upIndexOfRightShiftedResult = binaraysearchasc(up_slice, bi2de_modified(rightShiftResult_up))
                dn_slice = dnStates[ nosOfState_dn[noOfup-1] : nosOfState_dn[noOfup] ]
                dnIndexOfRightShiftedResult = binaraysearchasc(dn_slice, bi2de_modified(rightShiftResult_dn))
                basisIndexOfRightShiftedResult = ( nosOfState_all[noOfup-1] +
                    upIndexOfRightShiftedResult * (nosOfState_dn[noOfup] - nosOfState_dn[noOfup-1]) +
                    dnIndexOfRightShiftedResult )
                phase = -(1 - 2 * ((np.sum(upSector[:n+1]) + np.sum(dnSector[:rightShiftedIndex]) + noOfup - 2) % 2))
                SOC_rows.append(basisIndexOfRightShiftedResult)
                SOC_cols.append(m)
                SOC_elems.append(phase * t_so)
        
        # --- Shifting for spin down (flipping spin down to spin up) ---
        for p in dnNonZero:
            leftShiftedIndex = (p - 1) % noOfSites
            if upSector[leftShiftedIndex] != 1:
                leftShiftResult_dn = dnSector.copy()
                leftShiftResult_up = upSector.copy()
                leftShiftResult_dn[p] = 0
                leftShiftResult_up[leftShiftedIndex] = 1
                # After flip, the up count increases by one.
                up_slice = upStates[ nosOfState_up[noOfup+1] : nosOfState_up[noOfup+2] ]
                upIndexOfLeftShiftedResult = binaraysearchasc(up_slice, bi2de_modified(leftShiftResult_up))
                dn_slice = dnStates[ nosOfState_dn[noOfup+1] : nosOfState_dn[noOfup+2] ]
                dnIndexOfLeftShiftedResult = binaraysearchasc(dn_slice, bi2de_modified(leftShiftResult_dn))
                basisIndexOfLeftShiftedResult = ( nosOfState_all[noOfup+1] +
                    upIndexOfLeftShiftedResult * (nosOfState_dn[noOfup+2] - nosOfState_dn[noOfup+1]) +
                    dnIndexOfLeftShiftedResult )
                phase = -(1 - 2 * ((np.sum(dnSector[:p+1]) + np.sum(upSector[:leftShiftedIndex]) + noOfup - 1) % 2))
                SOC_rows.append(basisIndexOfLeftShiftedResult)
                SOC_cols.append(m)
                SOC_elems.append(phase * t_so)
            
            rightShiftedIndex = (p + 1) % noOfSites
            if upSector[rightShiftedIndex] != 1:
                rightShiftResult_dn = dnSector.copy()
                rightShiftResult_up = upSector.copy()
                rightShiftResult_dn[p] = 0
                rightShiftResult_up[rightShiftedIndex] = 1
                up_slice = upStates[ nosOfState_up[noOfup+1] : nosOfState_up[noOfup+2] ]
                upIndexOfRightShiftedResult = binaraysearchasc(up_slice, bi2de_modified(rightShiftResult_up))
                dn_slice = dnStates[ nosOfState_dn[noOfup+1] : nosOfState_dn[noOfup+2] ]
                dnIndexOfRightShiftedResult = binaraysearchasc(dn_slice, bi2de_modified(rightShiftResult_dn))
                basisIndexOfRightShiftedResult = ( nosOfState_all[noOfup+1] +
                    upIndexOfRightShiftedResult * (nosOfState_dn[noOfup+2] - nosOfState_dn[noOfup+1]) +
                    dnIndexOfRightShiftedResult )
                phase = (1 - 2 * ((np.sum(dnSector[:p+1]) + np.sum(upSector[:rightShiftedIndex]) + noOfup - 1) % 2))
                SOC_rows.append(basisIndexOfRightShiftedResult)
                SOC_cols.append(m)
                SOC_elems.append(phase * t_so)
    
    SOC_rows = np.array(SOC_rows)
    SOC_cols = np.array(SOC_cols)
    SOC_elems = np.array(SOC_elems)
    SOCHamiltonian = sp.csr_matrix((SOC_elems, (SOC_rows, SOC_cols)),
                                   shape=(totalNoOfPossiblestates, totalNoOfPossiblestates))
    
    totalHamiltonian = kineticHamiltonian + potentialHamiltonian + SOCHamiltonian + mzHamiltonian
    return totalHamiltonian, kineticHamiltonian, potentialHamiltonian, mzHamiltonian, SOCHamiltonian



def H_hubbard_soc(args):
    """
    Helper function to construct and return the dense Hamiltonian matrix for a given mz.
    
    Parameters:
      args : tuple containing (mz_val, t_s, alpha, t_so, U, N, N_par)
      
    Returns:
      H_dense: Dense Hamiltonian matrix as a NumPy array.
    """
    mz_val, t_s, alpha, t_so, U, N, N_par = args
    # Compute the Hamiltonian using your existing function.
    # The underscore variables are placeholders for unused outputs.
    H, _, _, _, _ = H_set_hubbard_soc(t_s, alpha, t_so, mz_val, U, N, N_par)
    
    # Convert to dense (assuming H is a sparse matrix) and return
    return H.toarray()

def Hubbard_SOC_eigenSystem_batch_alphaOff(t_s, t_so, U, mz_vals, N, N_par):
    """
    Builds and diagonalizes the Hamiltonian for each mz in mz_vals.
    No reciprocity term (alpha = 0), Hamiltonian is Hermitian.
    
    This version uses multiprocessing to construct the Hamiltonians in parallel,
    then performs a batched eigenvalue decomposition using TensorFlow.
    
    Parameters:
      t_s, t_so, U, N, N_par : model parameters
      mz_vals   : iterable of mz values
      N_par      : total number of particles
    """
    # Prepare the arguments for each process.
    args_list = [(mz_val, t_s, 0., t_so, U, N, N_par) for mz_val in mz_vals]

    # Use multiprocessing Pool to construct Hamiltonians in parallel.
    with mp.dummy.Pool() as pool:
        hamiltonians = pool.map(H_hubbard_soc, args_list)

    # Stack the Hamiltonian matrices into a 3D NumPy array with shape (num_mz, n, n)
    H_array = np.asarray(hamiltonians)
    
    # Convert the array to a TensorFlow tensor.
    H_batch = tf.convert_to_tensor(H_array, dtype=tf.float64)
    
    # Perform the batched eigenvalue decomposition.
    eigvals_batch, eigvecs_batch = tf.linalg.eigh(H_batch)
    
    # Convert the results back to NumPy arrays.
    eigvals_batch = eigvals_batch.numpy()
    eigvecs_batch = eigvecs_batch.numpy()

    # topological invariant
    y = np.asarray(mz_vals) < (2 * t_s)
    
    return eigvals_batch, eigvecs_batch, H_array, y.astype(int)



# ------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------
if __name__ == "__main__":
    noOfSites = 3
    noOfPar   = 2
    t_s   = 1.0
    alpha = 0.2
    t_so  = 0.1
    mz    = 0.05
    U     = 4.0

    (Htot, Hkin, Hpot, Hmz, Hsoc) = H_set_hubbard_soc(
        t_s, alpha, t_so, mz, U, noOfSites, noOfPar
    )

    print("Htot shape:", Htot.shape)
    print("Htot nnz:", Htot.nnz)
    print("Hkin nnz:", Hkin.nnz)
    print("Hpot nnz:", Hpot.nnz)
    print("Hmz  nnz:", Hmz.nnz)
    print("Hsoc nnz:", Hsoc.nnz)