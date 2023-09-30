import pandas as pd
import os

def get_video_files_from_path(path):
    #video extensions to search for
    video_extensions = ('.mp4', '.avi', '.MOV', '.mov')
    list_video_files = [filename for filename in os.listdir(path) if filename.endswith(video_extensions)]
    return list_video_files

def get_file_names_from_all_pickle(path):
    # List to store all filenames
    all_filenames = []
    # Iterate over all files in the directory
    for file in os.listdir(path):
        # Check if the file is a pickle file
        if file.endswith('.pkl'):
            # Construct the full path to the pickle file
            pickle_file_path = os.path.join(path, file)
            
            # Load the DataFrame from the pickle file
            video_files_df = pd.read_pickle(pickle_file_path)
            
            # Extract filenames from the DataFrame and add to the list
            filenames = video_files_df.iloc[:, 0].tolist()
            all_filenames.extend(filenames)

    return all_filenames

def is_video_processed(current_title, processed_video_names):
    # returns boolean, "if current_title in processed_vf..."
    return current_title in processed_video_names

# Example usage:
# path_to_pkl_directory = 'VideoFiles\\'
# all_video_names = get_file_names_from_all_pickle(path_to_pkl_directory)
# print(all_video_names)
