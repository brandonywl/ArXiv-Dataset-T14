import pathlib
import os

def get_root_folder():
    current_folder = pathlib.Path(__file__).parent.resolve()
    root_folder = os.path.join(current_folder, "..", "..")
    return root_folder

def get_folder_path(*recursive_to_join, root_folder=None):
    root_folder = root_folder if root_folder is not None else get_root_folder()
    data_folder = os.path.join(root_folder, *recursive_to_join)
    return data_folder

def get_archive_path(*recursive_to_join):
    recursive_to_join = ('archive',) + recursive_to_join
    return get_folder_path(*recursive_to_join)

def get_data_path(*recursive_to_join):
    recursive_to_join = ('data',) + recursive_to_join
    return get_folder_path(*recursive_to_join)

def get_src_path(*recursive_to_join):
    recursive_to_join = ('src',) + recursive_to_join
    return get_folder_path(recursive_to_join)