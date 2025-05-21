from mpi4py import MPI
import numpy as np

def file_processing_worker(lines_chunk, comm):
    # Count words and unique words locally
    words = []
    for line in lines_chunk:
        words.extend(line.strip().split())
    total_words_local = len(words)
    unique_words_local = set(words)
    
    # Reduce total words globally
    total_words = comm.reduce(total_words_local, op=MPI.SUM, root=0)
    
    # Gather all unique words sets at root
    unique_words_all = comm.gather(unique_words_local, root=0)
    
    if comm.rank == 0:
        # Combine all unique sets
        unique_words_global = set()
        for s in unique_words_all:
            unique_words_global.update(s)
        return total_words, len(unique_words_global)
    else:
        return None

def master_process_file(filename, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Read all lines in root
    if rank == 0:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    else:
        lines = None

    # Broadcast total number of lines
    num_lines = len(lines) if rank == 0 else None
    num_lines = comm.bcast(num_lines, root=0)

    # Determine counts and displs for scattering lines
    counts = [num_lines // size + (1 if i < num_lines % size else 0) for i in range(size)]
    displs = [sum(counts[:i]) for i in range(size)]

    # Scatter lines as objects (using Scatterv for arrays is complicated for strings,
    # so we do manual scatter via send/recv or Scatter with pickle enabled)
    # mpi4py Scatter can scatter Python objects automatically with root=0
    local_lines = comm.scatter([lines[displs[i]:displs[i]+counts[i]] for i in range(size)] if rank == 0 else None, root=0)

    # Process locally
    result = file_processing_worker(local_lines, comm)

    if rank == 0:
        total_words, unique_words_count = result
        print(f"Total words in file: {total_words}")
        print(f"Number of unique words: {unique_words_count}")

def file_process():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    filename = 'mpi_tasks/sample.txt'  # Change to your file path

    if rank == 0:
        print(f"Processing file: {filename}")

    master_process_file(filename, comm)

