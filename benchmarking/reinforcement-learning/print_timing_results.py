"""Print results."""
import glob
import os

import numpy as np

# Top-level directory containing all timing subdirectories
top_dir = "timing_results"
subdirs = [d for d in os.listdir(top_dir) if os.path.isdir(os.path.join(top_dir, d))]

# Collect all .npy files in the current directory
for subdir in sorted(subdirs):
    files = sorted(glob.glob(f"{top_dir}/{subdir}/*.npy"))

    # Dictionary to hold data
    data = {}

    # Load all files
    for file in files:
        name = file.replace(".npy", "").replace(f"{top_dir}/{subdir}/", "")
        data[name] = np.load(file)

    # Ensure all arrays are the same length
    lengths = [len(arr) for arr in data.values()]
    assert len(set(lengths)) == 1, "Arrays have different lengths!"
    N = lengths[0]

    def process_keyname(k):
        return (
            k.replace("TFRT_CPU_0", "cpu")
            .replace("cuda:0", "gpu")
            .replace("cuda", "gpu")
            .replace("gpu:0", "gpu")
        )

    headers = [process_keyname(k) for k in data.keys()]

    # Print header
    # print("\t".join(headers))

    # Print each row
    # for i in range(N):
    #     row = [f"{data[key][i]:.6f}" for key in data]
    #     print("\t".join(row))

    # print('\n\n')

    output_file = os.path.join(top_dir, "timing_summary.txt")
    with open(output_file, "a") as f:
        f.write(f"=== Results from {subdir} ===\n")
        f.write("\t".join(headers) + "\n")

        for i in range(N):
            row = [f"{data[key][i]:.6f}" for key in data]
            f.write("\t".join(row) + "\n")

print(f"results written to: {output_file}")
