from mpi4py import MPI

def count_all_occurrences_in_chunk(chunk_lines, keyword, case_sensitive=True):
    count = 0
    if not case_sensitive:
        keyword = keyword.lower()
    for line in chunk_lines:
        search_line = line if case_sensitive else line.lower()
        count += search_line.count(keyword)
    return count

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    filename = "mpi_tasks/large-text.txt"
    keyword = "God"
    case_sensitive = False

    if rank == 0:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        n_lines = len(lines)
        counts = [n_lines // size + (1 if i < n_lines % size else 0) for i in range(size)]
        displs = [sum(counts[:i]) for i in range(size)]

        chunks = [lines[displs[i]:displs[i] + counts[i]] for i in range(size)]
    else:
        chunks = None

    chunk = comm.scatter(chunks, root=0)

    local_count = count_all_occurrences_in_chunk(chunk, keyword, case_sensitive)

    total_count = comm.reduce(local_count, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"Total occurrences of '{keyword}': {total_count}")

if __name__ == "__main__":
    main()
