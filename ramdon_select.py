import os
import random
import shutil

# Define source and destination directories
src_dir = "dataset/images/lsb"
dst_dir = "dataset/images/lsb/select"

# Make sure the destination directory exists
os.makedirs(dst_dir, exist_ok=True)

# List all files in the source directory
all_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

# Randomly select 100 files
selected_files = random.sample(all_files, 100)

# Move selected files
for filename in selected_files:
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, filename)
    shutil.move(src_path, dst_path)

print(f"Moved {len(selected_files)} files to '{dst_dir}'.")
