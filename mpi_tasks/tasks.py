import sys
from mpi4py import MPI
import numpy as np
from PIL import Image
import pandas as pd
import time

# --- File Processing Task ---
def file_process_task(file):
    starttime = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # Your full code from file-process.py goes here

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

    def file_process(file):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        filename = 'mpi_tasks/sample.txt'  # Change to your file path

        if rank == 0:
            print(f"Processing file: {file}")

        master_process_file(file, comm)


    if rank == 0:
        print("File processing task running")
    file_process(file)
    # ... rest of the code ...
    end_time = time.time()

    elapsed = end_time - starttime
    print(f"Elapsed time: {elapsed:.4f} seconds")

# --- Image Processing Task ---
def image_process_task(file):
    starttime = time.time()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    def apply_grayscale(image_chunk):
        # Convert RGB chunk to grayscale using luminosity method
        # image_chunk shape: (rows, cols, 3)
        gray = (0.2989 * image_chunk[:,:,0] + 0.5870 * image_chunk[:,:,1] + 0.1140 * image_chunk[:,:,2]).astype(np.uint8)
        # Convert grayscale single channel to 3-channel grayscale image
        gray_3ch = np.stack((gray,)*3, axis=-1)
        return gray_3ch

    def apply_blur(image_chunk, iterations=3):
        kernel_size = 3
        pad = kernel_size // 2
        blurred = image_chunk.copy()
        for _ in range(iterations):
            padded = np.pad(blurred, ((pad,pad),(pad,pad),(0,0)), mode='edge')
            temp = np.zeros_like(blurred)
            rows, cols, _ = blurred.shape
            for i in range(rows):
                for j in range(cols):
                    region = padded[i:i+kernel_size, j:j+kernel_size]
                    temp[i,j] = np.mean(region, axis=(0,1)).astype(np.uint8)
            blurred = temp
        return blurred

    def master_process(filename):
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        # Load image in root
        img = Image.open(filename).convert('RGB')
        img_np = np.array(img)
        height, width, channels = img_np.shape

        # Split image rows among processes
        counts = [height // size + (1 if i < height % size else 0) for i in range(size)]
        displs = [sum(counts[:i]) for i in range(size)]

        # Send counts of rows * width * channels
        send_counts = [c * width * channels for c in counts]
        send_displs = [sum(send_counts[:i]) for i in range(size)]

        # Flatten image array for Scatterv
        flat_img = img_np.flatten() if rank == 0 else None

        # Allocate buffer for local chunk
        local_size = send_counts[rank]
        local_flat = np.zeros(local_size, dtype=np.uint8)

        comm.Scatterv([flat_img, send_counts, send_displs, MPI.UNSIGNED_CHAR], local_flat, root=0)

        # Reshape local chunk to 3D
        local_rows = counts[rank]
        local_img = local_flat.reshape((local_rows, width, channels))

        # Apply filter
        gray_chunk = apply_grayscale(local_img)
        blur_chunk = apply_blur(local_img,5)

        # Flatten both processed chunks
        gray_flat = gray_chunk.flatten()
        blur_flat = blur_chunk.flatten()

        if rank == 0:
            recv_gray = np.zeros_like(flat_img)
            recv_blur = np.zeros_like(flat_img)
        else:
            recv_gray = None
            recv_blur = None

        comm.Gatherv(gray_flat, [recv_gray, send_counts, send_displs, MPI.UNSIGNED_CHAR], root=0)
        comm.Gatherv(blur_flat, [recv_blur, send_counts, send_displs, MPI.UNSIGNED_CHAR], root=0)

        if rank == 0:
            gray_img = recv_gray.reshape((height, width, channels))
            blur_img = recv_blur.reshape((height, width, channels))

            Image.fromarray(gray_img).save(f"images/processed_grayscale_{str(file).split('/')[1].split('.')[0]}.png")
            Image.fromarray(blur_img).save(f"images/processed_blur_{str(file).split('/')[1].split('.')[0]}.png")

            print("Saved processed_grayscale.png and processed_blur.png")

    filename = 'mpi_tasks/N.jpeg'  # Replace with your image path

    # Broadcast filename and filter_type to all processes if needed
    # For simplicity, assume same file path known on all processes

    if rank == 0:
        print("Image processing task running")
    master_process(file)
    # ... rest of the code ...
    end_time = time.time()

    elapsed = end_time - starttime
    print(f"Elapsed time: {elapsed:.4f} seconds")

# --- Large Text Search Task ---
def large_text_task(file,keyword):
    starttime = time.time()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    def count_all_occurrences_in_chunk(chunk_lines, keyword, case_sensitive=True):
        count = 0
        if not case_sensitive:
            keyword = keyword.lower()
        for line in chunk_lines:
            search_line = line if case_sensitive else line.lower()
            count += search_line.count(keyword)
        return count
    filename = "mpi_tasks/large-text.txt"

    case_sensitive = False
    if rank == 0:
        print("Large text search task running")
        
    if rank == 0:
        with open(file, 'r', encoding='utf-8') as f:
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

    # ... rest of the code ...
    end_time = time.time()

    elapsed = end_time - starttime
    print(f"Elapsed time: {elapsed:.4f} seconds")

# --- MPI Machine Learning Task ---
def mpi_ml_task(filename, target_col):
    starttime = time.time()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
        
    def load_csv_with_pandas(filename, target_col):
        df = pd.read_csv(filename)
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")
        X_df = df.drop(columns=[target_col])
        X_df.insert(0, 'Intercept', 1)  # Add intercept column (raw features)
        y_df = df[target_col]
        X = X_df.to_numpy(dtype=np.float64)
        y = y_df.to_numpy(dtype=np.float64)
        return X, y

    def parallel_linear_regression(X, y, comm, lambda_reg=1e-5):
        rank = comm.Get_rank()
        size = comm.Get_size()

        n_samples, n_features = X.shape

        counts = [n_samples // size + (1 if i < n_samples % size else 0) for i in range(size)]
        displs = [sum(counts[:i]) for i in range(size)]

        counts_elements = np.array([c * n_features for c in counts], dtype=int)
        displs_elements = np.array([d * n_features for d in displs], dtype=int)

        local_rows = counts[rank]
        local_X = np.empty((local_rows, n_features), dtype=np.float64, order='C')
        local_y = np.empty(local_rows, dtype=np.float64)

        if rank == 0:
            flat_X = np.ascontiguousarray(X).ravel()
        else:
            flat_X = None

        comm.Scatterv([flat_X, counts_elements, displs_elements, MPI.DOUBLE], local_X.ravel(), root=0)
        comm.Scatterv([y if rank == 0 else None, np.array(counts, dtype=int), np.array(displs, dtype=int), MPI.DOUBLE], local_y, root=0)

        XtX_local = local_X.T @ local_X
        Xty_local = local_X.T @ local_y


        XtX = np.zeros((n_features, n_features), dtype=np.float64)
        Xty = np.zeros(n_features, dtype=np.float64)

        comm.Reduce(XtX_local, XtX, op=MPI.SUM, root=0)
        comm.Reduce(Xty_local, Xty, op=MPI.SUM, root=0)

    
        intercept = None
        coeffs = None
        if rank == 0:
            XtX_reg = XtX + lambda_reg * np.eye(n_features)
            theta = np.linalg.solve(XtX_reg, Xty)
            intercept = theta[0]
            coeffs = theta[1:]

        return intercept, coeffs

    def predict(X_raw, intercept, coeffs):
        n_samples = X_raw.shape[0]
        X_with_intercept = np.hstack((np.ones((n_samples, 1)), X_raw))
        return X_with_intercept @ np.hstack((intercept, coeffs))
    
    if rank == 0:
        print(f"Loading data from {filename} without scaling")
        X, y = load_csv_with_pandas(filename, target_col)
        # print("Sample X (with intercept):")
        # print(X[:5])
        # print("Sample y:")
        # print(y[:5])
    else:
        X, y = None, None

    n_samples = comm.bcast(X.shape[0] if rank == 0 else None, root=0)
    n_features = comm.bcast(X.shape[1] if rank == 0 else None, root=0)

    if rank != 0:
        X = np.empty((n_samples, n_features), dtype=np.float64)
        y = np.empty(n_samples, dtype=np.float64)

    intercept, coeffs = parallel_linear_regression(X, y, comm)

    
    if rank == 0:
        print("ML training task running")
    if rank == 0:
        print(f"Intercept: {intercept}")
        print(f"Coefficients: {coeffs}")

        # Example prediction on new data without intercept column
        X_test_raw = np.array([
            [2000, 3, 15],
            [1500, 2, 10]
        ])
        preds = predict(X_test_raw, intercept, coeffs)
        print(f"Predictions: {preds}")
    # ... rest of the code ...
    end_time = time.time()

    elapsed = end_time - starttime
    print(f"Elapsed time: {elapsed:.4f} seconds")

# --- Odd-Even Sort Task ---
def odd_even_sort_task(data):
    starttime = time.time()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
        
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

    def master_process(data):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # Prepare data

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


    if rank == 0:
        print("Odd-even sort task running")
    n = None
    if rank == 0:
        n = 40
    n = comm.bcast(n, root=0)
    
    counts = np.array([n // comm.Get_size() + (1 if i < n % comm.Get_size() else 0) for i in range(comm.Get_size())], dtype=int)
    displs = np.array([sum(counts[:i]) for i in range(comm.Get_size())], dtype=int)
    
    if rank == 0:
        master_process(data)
    else:
        worker_process(comm, counts, displs, n)
    # ... rest of the code ...
    end_time = time.time()

    elapsed = end_time - starttime
    print(f"Elapsed time: {elapsed:.4f} seconds")

# --- Statistics Analyzer Task ---
def statics_ana_task(filename,target_col):
    starttime = time.time()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
        
    def local_stats(data_chunk):
        n = len(data_chunk)
        s = np.sum(data_chunk)
        ss = np.sum(data_chunk**2)
        minimum = np.min(data_chunk)
        maximum = np.max(data_chunk)
        return n, s, ss, minimum, maximum

    if rank == 0:
        print("Statistics analysis task running")
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
    # ... rest of the code ...
    end_time = time.time()

    elapsed = end_time - starttime
    print(f"Elapsed time: {elapsed:.4f} seconds")

# --- Distributed Matrix Multiplication Task ---
def dist_mat_mult_task(MatA,MatB):
    starttime = time.time()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

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
    if rank == 0:
        # Read matrices from CSV files
        A_df = pd.read_csv(MatA, header=None)  # no headers, adjust if needed
        B_df = pd.read_csv(MatB, header=None)
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
    end_time = time.time()
    elapsed = end_time - starttime
    print(f"Elapsed time: {elapsed:.4f} seconds")
# --- Dispatcher ---
def main():
    if len(sys.argv) < 2:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Usage: mpirun -np <num> python all_tasks.py <task_name>")
            print("Tasks: file_process, image_process, large_text, mpi_ml, odd_even_sort, statics_ana, dist_mat_mult")
        sys.exit(1)
    
    task = sys.argv[1].lower()
    

    
    # print(files)
    
    if task == "file_process":
        if len(sys.argv)>2:
            
            file1 = sys.argv[2]

            if file1 == None:
                file_process_task('mpi_tasks/sample.txt')
            else:
                file_process_task(file1)
        else:
            file_process_task('mpi_tasks/sample.txt')
    elif task == "image_process":
        if len(sys.argv)>2:
            
            file1 = sys.argv[2]

            if file1 == None:
                image_process_task('mpi_tasks/N.jpeg')
            else:
                image_process_task(file1)
        else:
            image_process_task('mpi_tasks/N.jpeg')
    elif task == "parallel_search":
        if len(sys.argv)>2:
            
            file1 = sys.argv[2]
            file2 = sys.argv[3]
            if file1 == None or file2 == None:
                large_text_task('mpi_tasks/large-text.txt','God')
            else:
                large_text_task(file1,file2)
        else:
            large_text_task('mpi_tasks/large-text.txt','God')
    elif task == "linear_regression":
        if len(sys.argv)>2:
                
            file1 = sys.argv[2]
            file2 = sys.argv[3]
            if file1 == None or file2 == None:
                mpi_ml_task('mpi_tasks/data.csv','Price')
            else:
                mpi_ml_task(file1,file2)
        else:
            mpi_ml_task('mpi_tasks/data.csv','Price')

    elif task == "sort":
        print(task)
        print(len(sys.argv))
        if len(sys.argv)>2:
            print("I am here")
            file1 = sys.argv[2]
            if file1 == None:
                odd_even_sort_task(np.random.randint(1, 1000, size=40).astype('i'))
            else:
                print(file1)
                data = str(file1).split(',')
                Data = np.array([float(s) for s in data])
                print(Data)
                # df = pd.read_csv(file1)
                # Data = df.to_numpy(dtype=float)
                # odd_even_sort_task(Data)
        else:
            
            odd_even_sort_task(np.random.randint(1, 1000, size=40).astype('i'))
       
            
        

    elif task == "statics-ana":
        if len(sys.argv)>2:
            
            file1 = sys.argv[2]
            file2 = sys.argv[3]
            if file1 == None or file2 == None:
                statics_ana_task("mpi_tasks/data.csv",'Price')
            else:
                statics_ana_task(file1,file2)
        else:
            statics_ana_task("mpi_tasks/data.csv",'Price')

    elif task == "matrix_multiply":
        if len(sys.argv)>2:
            file1 = sys.argv[2]
            file2 = sys.argv[3]
            if file1 == None or file2 == None:
                dist_mat_mult_task('mpi_tasks/matrixA.csv','mpi_tasks/matrixB.csv')
            else:
                dist_mat_mult_task(file1,file2)
        else:
            dist_mat_mult_task('mpi_tasks/matrixA.csv','mpi_tasks/matrixB.csv')

    else:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"Unknown task: {task}")
        sys.exit(1)

if __name__ == "__main__":
    main()
