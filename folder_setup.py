# setup_folders.py
import os

# The main directory where all school data will be stored
DATA_DIR = "data"

# A list of all the CUNY school names (use short, lowercase names for folders)
CUNY_SCHOOLS = [
    'baruch', 'bmcc', 'bronxcc', 'brooklyn', 'citytech', 'csi', 'guttman',
    'hostos', 'hunter', 'johnjay', 'kingsborough', 'laguardia', 'lehman',
    'medgarevers', 'queens', 'queensborough', 'york', 'ccny',
    'cunygrad', 'cunylaw', 'cunysph', 'cunyslu', 'cunysps'
]

# --- NEW: A list of subfolders to create within each school's folder ---
# You can add more subfolder names here in the future, like 'schedules'
SUBFOLDERS = [
    'rmp_reviews'
]

def create_school_folders():
    """
    Creates the main data directory and a subfolder for each CUNY school.
    Also creates specified subfolders within each school folder.
    """
    print("Setting up CUNY data folders...")
    
    # Create the main 'data' directory if it doesn't exist
    # --- THE FIX IS HERE: Corrected the typo from 'makdirs' to 'makedirs' ---
    os.makedirs(DATA_DIR, exist_ok=True)

    # Loop through the list of schools and create a folder for each one
    for school in CUNY_SCHOOLS:
        school_path = os.path.join(DATA_DIR, school)
        os.makedirs(school_path, exist_ok=True)
        print(f"  - Created folder: {school_path}")

        # --- NEW: Create subfolders inside the school folder ---
        for subfolder in SUBFOLDERS:
            subfolder_path = os.path.join(school_path, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)
            print(f"    - Created subfolder: {subfolder_path}")


    print("\n✅ Folder setup complete!")

# This makes the script runnable from the command line
if __name__ == "__main__":
    create_school_folders()

