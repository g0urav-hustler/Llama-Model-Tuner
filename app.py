import os
import time
import yaml
import streamlit as st
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_processing import DataProcessing
from src.components.train_model import ModelTrain
from transformers import AutoTokenizer




@st.cache_resource(show_spinner="Loading model tokenizer...")
def load_tokenizer( model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name= model_path,)
    return tokenizer


@st.cache_resource(show_spinner="Loading model for testing..")
def load_model( model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_name= model_path,)
    return model


# page config
st.set_page_config(page_title="LLama Model Tuner", layout="wide")

# title
st.title("LLama Model Tuner Application")

#subtitle
st.markdown("This application helps you to fine llama model on your custom dataset. ")

uploaded_file = st.file_uploader("Upload your data set file in csv format")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    df_columns = list(df.columns)
    
    df_columns.insert(0, "None")

    with st.form("my_form"):

        #subtitle
        st.markdown("Enter the data from dataset")
        col1, col2 = st.columns(2)
        
        # context
        with col1:
            question_column = st.selectbox(label = "Select the question column", options= df_columns)

        # question column input
        with col2:
            answer_column = st.selectbox(label = "Select the answer column", options= df_columns)

        train_data_range = list(range(10,100,10))
        data_size_col1, data_size_col2, data_size_col3 = st.columns(3)

        # train_data_size input
        # train_data_range = 80 
        with data_size_col1:
            train_data_size = st.selectbox("Enter training data size ", options=train_data_range)

        # val_data_size input
        validation_data_range = list(range(10,100,10))
        with data_size_col2:
            val_data_size = st.selectbox("Enter validation data size ", options= validation_data_range)

        test_data_range = list(range(0,100,10))
        with data_size_col3:
            test_data_size = st.selectbox("Enter test data size ", options= test_data_range)

        st.markdown("Select the model parameter")
        
        par_col1, par_col2, par_col3, par_col4 = st.columns(4)

        # model type input
        with par_col1:
            model_name = st.text_input(label="Enter model name ", value = None)

        # epochs input
        with par_col2:
            no_of_epochs = st.number_input("Enter the no of epochs", value=0)

        # train batch size
        with par_col3:
            train_batch_size = st.number_input("Enter train batch size ", value = 4)

        # val batch size input
        with par_col4:
            val_batch_size = st.number_input("Enter val batch size", value = 4)

        sb_col1, sb_col2, sb_col3, sb_col4, sb_col5 = st.columns(5)
        with sb_col3:
            submitted = st.form_submit_button(label = "Submit Parameters")


    if submitted:

        params_data = {
        "data_processing":
            {
                "question_col": question_column,
                "answer_col": answer_column,
                
                "train_data_size": train_data_size / 100,
                "val_data_size": val_data_size /100 ,
                
            },

        "model_params":
            {
                "model_name": model_name,
                "epochs": no_of_epochs,
                "train_batch_size": train_batch_size,
                "val_batch_size": val_batch_size
            }

        }

        folder = 'web_files'
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, "data_file.csv")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with open('params.yaml', 'w') as outfile:
            yaml.dump(params_data, outfile, default_flow_style=False)

    button_col1, button_col2, button_col3, button_col4, button_col5 = st.columns(5)
    if os.path.isfile("params.yaml"):
        with button_col3:
            train_button = st.button("Train The Model")
        if train_button and not submitted:
            with st.status("Training Status", expanded=True) as status:
                st.write("Data Loading.")
                data_ingestion_process = DataIngestion()
                data_ingestion_process.get_raw_data()
                time.sleep(3)
                st.write("Data Processing")
                data_processing_process = DataProcessing()
                data_processing_process.get_processed_data()
                data_processing_process.get_split_data()
                time.sleep(2)
                st.write("Training model..")
                train_model_process = ModelTrain()
                training_result = train_model_process.train_model()
                status.update(label="Model Trained Succesfully", state="complete", expanded=False)
 
            st.success('Done!')

    if os.path.isdir(os.path.join("artifacts","models",model_name)):

        if training_result != None:

            result_col1, result_col2 = st.columns(2)
                
            with result_col1:
                st.write("Model Parameters")
                st.write(params_data["model_params"])

            with result_col2:
                
                st.write(training_result)
                    
        st.write("Try Trained Model Output")

        tokenizer = load_tokenizer(os.path.join("artifacts","models",model_name, "tokenizer"))
        model = load_model(os.path.join("artifacts","models",model_name, "model"))

        input_prompt = st.text_input("Try any question.. ",None, placeholder= "Write question here..")
        if input_prompt:
            st.write("Prediction")
            pipe = pipeline(task="text-generation", model= model, tokenizer= tokenizer, max_length=200)
            result = pipe(f"[INST] {input_prompt} [/INST]")
            print(result[0]['generated_text'])
            answer = model.predict()
            st.write("Predicted Answer")
            st.write(answer)


        with open(f"{model_name}.zip", "rb") as fp:
            btn = st.download_button(
                label="Download Trained Model",
                data=fp,
                file_name=f"{model_name}.zip",
                mime="application/octet-stream"
                )
            
footer = """<style>.footer {position: fixed;left: 0;bottom: 0;width: 100%;background-color: #000;color: white;text-align: center;}
</style><div class='footer'><p>Made By Gourav Chouhan</p></div>"""
st.markdown(footer, unsafe_allow_html=True)

