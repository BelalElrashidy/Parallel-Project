from mpi4py import MPI
import numpy as np
import pandas as pd

def master(A, B, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    m, n = A.shape
    n_b, p = B.shape
    assert n == n_b, "Inner matrix dimensions must match"

    # Calculate counts and displacements for rows of A
    counts = [m // size + (1 if i < m % size else 0) for i in range(size)]
    displs = [sum(counts[:i]) for i in range(size)]

    # Convert to elements (flattened arrays)
    counts_A = np.array([c * n for c in counts], dtype=int)
    displs_A = np.array([d * n for d in displs], dtype=int)

    counts_C = np.array([c * p for c in counts], dtype=int)
    displs_C = np.array([d * p for d in displs], dtype=int)

    # Broadcast matrix B
    flat_B = np.ascontiguousarray(B).ravel()
    comm.Bcast(flat_B, root=0)
    B = flat_B.reshape(n, p)

    # Scatter rows of A
    local_rows = counts[0]
    local_A = np.empty((local_rows, n), dtype=np.float64)
    comm.Scatterv([np.ascontiguousarray(A).ravel(), counts_A, displs_A, MPI.DOUBLE], local_A.ravel(), root=0)

    # Compute local multiplication
    local_C = local_A @ B

    # Prepare full matrix C to gather results
    C = np.empty((m, p), dtype=np.float64)
    comm.Gatherv(local_C.ravel(), [C.ravel(), counts_C, displs_C, MPI.DOUBLE], root=0)

    return C

def worker(comm, counts, displs, n, p):
    rank = comm.Get_rank()

    counts_A = np.array([c * n for c in counts], dtype=int)
    displs_A = np.array([d * n for d in displs], dtype=int)

    counts_C = np.array([c * p for c in counts], dtype=int)
    displs_C = np.array([d * p for d in displs], dtype=int)

    local_rows = counts[rank]
    local_A = np.empty((local_rows, n), dtype=np.float64, order='C')
    comm.Scatterv([None, counts_A, displs_A, MPI.DOUBLE], local_A.ravel(), root=0)

    flat_B = np.empty(n * p, dtype=np.float64)
    comm.Bcast(flat_B, root=0)
    B = flat_B.reshape(n, p)

    local_C = local_A @ B

    comm.Gatherv(local_C.ravel(), [None, counts_C, displs_C, MPI.DOUBLE], root=0)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Read matrices from CSV files
        A_df = pd.read_csv('mpi_tasks/matrixA.csv', header=None)  # no headers, adjust if needed
        B_df = pd.read_csv('mpi_tasks/matrixB.csv', header=None)
        A = A_df.to_numpy(dtype=np.float64)
        B = B_df.to_numpy(dtype=np.float64)


        m, n = A.shape
        n_b, p = B.shape
        assert n == n_b

        counts = [m // size + (1 if i < m % size else 0) for i in range(size)]
        displs = [sum(counts[:i]) for i in range(size)]

        counts = comm.bcast(counts, root=0)  # broadcast Python list
        displs = comm.bcast(displs, root=0)  # broadcast Python list
        n = comm.bcast(n, root=0)
        p = comm.bcast(p, root=0)

        C = master(A, B, comm)

        print("Result C = A x B:")
        print(C)

    else:
        counts = comm.bcast(None, root=0)  # receive Python list
        displs = comm.bcast(None, root=0)  # receive Python list
        n = comm.bcast(None, root=0)
        p = comm.bcast(None, root=0)
        worker(comm, counts, displs, n, p)

if __name__ == "__main__":
    main()
