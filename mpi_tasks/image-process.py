from mpi4py import MPI
from PIL import Image
import numpy as np

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

        Image.fromarray(gray_img).save('processed_grayscale.png')
        Image.fromarray(blur_img).save('processed_blur.png')

        print("Saved processed_grayscale.png and processed_blur.png")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    filename = 'mpi_tasks/N.jpeg'  # Replace with your image path
    filter_type = 'grayscale'    # or 'blur'

    if rank == 0:
        print(f"Applying {filter_type} filter to {filename}")

    # Broadcast filename and filter_type to all processes if needed
    # For simplicity, assume same file path known on all processes

    master_process(filename)

if __name__ == "__main__":
    main()
