from mpi4py import MPI
import numpy as np
import pandas as pd

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

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    filename = 'mpi_tasks/data.csv'      # your CSV file path
    target_col = 'Price'       # your target column name

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
        print(f"Intercept: {intercept}")
        print(f"Coefficients: {coeffs}")

        # Example prediction on new data without intercept column
        X_test_raw = np.array([
            [2000, 3, 15],
            [1500, 2, 10]
        ])
        preds = predict(X_test_raw, intercept, coeffs)
        print(f"Predictions: {preds}")

if __name__ == "__main__":
    main()
