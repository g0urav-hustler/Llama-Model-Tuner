import os
import pandas as pd
from src.constants import *
from src.utils.common import read_yaml,create_directories, save_json


class DataProcessing():

    def __init__(self):
        self.config = read_yaml(CONFIG_FILE_PATH)
        self.params = read_yaml(PARAMS_FILE_PATH)
        create_directories([self.config.data_processing.processed_data_dir])

    def create_formated_data(self, questions_list, answers_list):

        text_list = []

        for i in range(len(questions_list)):
            text = f"<s> [INST] {questions_list[i]} [/INST] {answers_list[i]} </s>"

            text_list.append(text)

        return {"text": text_list}
        
    def get_processed_data(self):
        config = self.config.data_processing
        params = self.params.data_processing
        file_path = os.path.join(config.raw_data_dir, "data_file.csv")

        df = pd.read_csv(file_path)

        formated_data = self.create_formated_data(df[params.question_col], df[params.answer_col])
        save_json(Path(os.path.join(config.processed_data_dir, "processed_data.json")), formated_data)



if __name__ == "__main__":
    obj = DataProcessing()
    obj.get_processed_data()
