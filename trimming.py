import os
import shutil

def prune_files(directory, target_file_count):
    for subdir, dirs, files in os.walk(directory):
        total_files = len(files)
        if total_files <= target_file_count:
            continue 
        

        files_to_delete = total_files - target_file_count
        full_file_paths = [os.path.join(subdir, file) for file in files]
        full_file_paths.sort(key=lambda x: os.path.getmtime(x))

        for file_path in full_file_paths[:files_to_delete]:
            try:
                os.remove(file_path)
                print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {str(e)}")

directory_path = './data'
prune_files(directory_path, 500)
