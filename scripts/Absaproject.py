import os

def create_folder_structure(base_dir='absa_project'):
    # Define the folder structure
    folders = [
        'data',
        'notebooks',
        'models',
        'reports',
        'scripts'
    ]
    
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create each folder in the structure
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")

if __name__ == "__main__":
    create_folder_structure()