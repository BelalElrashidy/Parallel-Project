from mpi4py import MPI
import numpy as np
import pandas as pd

def local_stats(data_chunk):
    n = len(data_chunk)
    s = np.sum(data_chunk)
    ss = np.sum(data_chunk**2)
    minimum = np.min(data_chunk)
    maximum = np.max(data_chunk)
    return n, s, ss, minimum, maximum

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    filename = "mpi_tasks/data.csv"
    target_col = "Price"  # numeric column to analyze

    if rank == 0:
        df = pd.read_csv(filename)
        if target_col not in df.columns:
            raise ValueError(f"Column '{target_col}' not found in CSV")
        data = df[target_col].to_numpy(dtype=np.float64)
        n = len(data)

        counts = [n // size + (1 if i < n % size else 0) for i in range(size)]
        displs = [sum(counts[:i]) for i in range(size)]
        chunks = [data[displs[i]:displs[i] + counts[i]] for i in range(size)]
    else:
        chunks = None

    # Scatter data chunks
    chunk = comm.scatter(chunks, root=0)

    # Compute local stats
    n_local, s_local, ss_local, min_local, max_local = local_stats(chunk)

    # Reduce sums and counts to root
    n_global = comm.reduce(n_local, op=MPI.SUM, root=0)
    s_global = comm.reduce(s_local, op=MPI.SUM, root=0)
    ss_global = comm.reduce(ss_local, op=MPI.SUM, root=0)
    min_global = comm.reduce(min_local, op=MPI.MIN, root=0)
    max_global = comm.reduce(max_local, op=MPI.MAX, root=0)

    if rank == 0:
        mean = s_global / n_global
        variance = (ss_global / n_global) - mean**2
        stddev = np.sqrt(variance)
        print(f"Count: {n_global}")
        print(f"Mean: {mean}")
        print(f"Variance: {variance}")
        print(f"Standard Deviation: {stddev}")
        print(f"Min: {min_global}")
        print(f"Max: {max_global}")

if __name__ == "__main__":
    main()
