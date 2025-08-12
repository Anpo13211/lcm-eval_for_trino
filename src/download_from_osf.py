import argparse
import os
import osfclient
from dotenv import load_dotenv
from tqdm import tqdm

from classes.paths import LocalPaths

if __name__ == "__main__":
    load_dotenv()
    output_path = LocalPaths().data
    output_path = os.path.expanduser(LocalPaths().data)
    os.makedirs(output_path, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--artifacts',
        choices=["runs", "workloads", "models", "evaluation", "datasets"],
        default=["runs"],
        nargs='+')
    args = parser.parse_args()

    # create connection to OSF
    osf = osfclient.OSF()
    project = osf.project("rb5tn")
    print(f'Download files from OSF Project "{project.title}" (ID: {project.id})')
    storage = project.storage()
    files = list(storage.files)

    for file in tqdm(files, desc="Downloading files", total=len(files)):
        file_path = file.path.strip("/")  # Remove leading slash for local path compatibility
        if not any(file_path.startswith(f"{artifact}") for artifact in args.artifacts):
            print(f"Skipping file: {file.path} (not in specified artifacts)")
            continue
        print(f"Downloading file: {file.path} (size: {file.size / 1000000:.3f} MB)")
        target_file_path = os.path.join(output_path, file_path)

        if os.path.exists(target_file_path):
            print(f"File {target_file_path} already exists, skipping download.")
            continue

        target_file_base_dir = os.path.dirname(target_file_path)

        os.makedirs(target_file_base_dir, exist_ok=True)

        with open(target_file_path, "wb") as f:
            file.write_to(f)

        # extract if it's a zip file
        if file_path.endswith(".zip"):
            import zipfile

            print("Unzipping...")
            with zipfile.ZipFile(target_file_path, "r") as zip_ref:
                zip_ref.extractall(target_file_base_dir)
            print(f"Extracted {file_path} to {target_file_base_dir}")

            # remove the zip file after extraction
            os.remove(target_file_path)

            # Check if the directory contains zips as well, unzuip them
            for root, dirs, files in os.walk(target_file_base_dir):
                for file in files:
                    if file.endswith(".zip"):
                        zip_file_path = os.path.join(root, file)
                        print(f"Unzipping {zip_file_path}...")
                        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                            zip_ref.extractall(root)
                        print(f"Extracted {zip_file_path} to {root}")
                        os.remove(zip_file_path)
    print(f"All files downloaded to: {output_path}")
