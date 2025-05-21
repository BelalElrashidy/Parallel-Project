from mpi4py import MPI
import numpy as np

def worker_process(comm, counts, displs, n):
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    local_n = counts[rank]
    local_data = np.zeros(local_n, dtype='i')
    
    # Receive chunk of data
    comm.Scatterv([None, counts, displs, MPI.INT], local_data, root=0)
    
    # Local sort
    local_data.sort()
    
    for phase in range(size):
        if phase % 2 == 0:
            partner = rank + 1 if rank % 2 == 0 else rank - 1
        else:
            partner = rank - 1 if rank % 2 == 0 else rank + 1
        
        if 0 <= partner < size:
            send_data = local_data.copy()
            recv_data = np.empty_like(local_data)
            
            comm.Sendrecv(send_data, dest=partner, sendtag=phase,
                          recvbuf=recv_data, source=partner, recvtag=phase)
            
            if rank < partner:
                combined = np.concatenate((local_data, recv_data))
                local_data = np.sort(combined)[:local_n]
            else:
                combined = np.concatenate((recv_data, local_data))
                local_data = np.sort(combined)[-local_n:]
    
    # Send sorted chunk back
    sorted_chunk = None
    if rank == 0:
        sorted_chunk = np.empty(n, dtype='i')
    comm.Gatherv(local_data, [sorted_chunk, counts, displs, MPI.INT], root=0)
    
    return sorted_chunk

def master_process():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Prepare data
    data = np.random.randint(1, 1000, size=40).astype('i')
    print("Original data:", data)
    
    n = len(data)
    
    counts = np.array([n // size + (1 if i < n % size else 0) for i in range(size)], dtype=int)
    displs = np.array([sum(counts[:i]) for i in range(size)], dtype=int)
    
    # Broadcast data length
    comm.bcast(n, root=0)
    
    # Scatter data
    local_n = counts[rank]
    local_data = np.zeros(local_n, dtype='i')
    comm.Scatterv([data, counts, displs, MPI.INT], local_data, root=0)
    
    # Local sort
    local_data.sort()
    
    # Odd-even phases (same as worker)
    for phase in range(size):
        if phase % 2 == 0:
            partner = rank + 1 if rank % 2 == 0 else rank - 1
        else:
            partner = rank - 1 if rank % 2 == 0 else rank + 1
        
        if 0 <= partner < size:
            send_data = local_data.copy()
            recv_data = np.empty_like(local_data)
            
            comm.Sendrecv(send_data, dest=partner, sendtag=phase,
                          recvbuf=recv_data, source=partner, recvtag=phase)
            
            if rank < partner:
                combined = np.concatenate((local_data, recv_data))
                local_data = np.sort(combined)[:local_n]
            else:
                combined = np.concatenate((recv_data, local_data))
                local_data = np.sort(combined)[-local_n:]
    
    # Gather sorted data
    sorted_data = np.empty(n, dtype='i')
    comm.Gatherv(local_data, [sorted_data, counts, displs, MPI.INT], root=0)
    
    print("Sorted data:", sorted_data)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    n = None
    if rank == 0:
        n = 40
    n = comm.bcast(n, root=0)
    
    counts = np.array([n // comm.Get_size() + (1 if i < n % comm.Get_size() else 0) for i in range(comm.Get_size())], dtype=int)
    displs = np.array([sum(counts[:i]) for i in range(comm.Get_size())], dtype=int)
    
    if rank == 0:
        master_process()
    else:
        worker_process(comm, counts, displs, n)


if __name__ == "__main__":
    main()
