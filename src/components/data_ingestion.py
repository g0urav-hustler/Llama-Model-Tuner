import os
from src.constants import *
from src.utils.common import read_yaml, copy_files, create_directories
from src.utils.common import copy_files


class DataIngestion():

    def __init__(self):
        self.config = read_yaml(CONFIG_FILE_PATH)
        create_directories([self.config.data_ingestion.raw_data_dir])
        
    def get_raw_data(self):
        source_dir = self.config.data_ingestion.web_data_dir
        target_dir = self.config.data_ingestion.raw_data_dir
        files_list = os.listdir(source_dir)
        
        # copy file to raw data
        copy_files(files_list, source_dir, target_dir, file_extension=False)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.get_raw_data()
