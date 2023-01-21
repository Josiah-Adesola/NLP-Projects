import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch


@st.cache(allow_output_mutation=True)

def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained("Josiah-Adesola/Bert-Streamlit-Finetune")
    return tokenizer, model

tokenizer, model = get_model()

st.title("Toxic Statements Text Classification Web App")
st.markdown("This project was developed by Josiah Adesola, and greatly refrenced from Pradip Nichite tutorial video ")
#(https://www.youtube.com/watch?v=mvIp9TvPMh0&list=PLAMHV77MSKJ4Z4OXqao1gRdfQK7VQYAXb&index=9&t=292s) 
st.markdown("This model was gotten from the pretrained bert model, fine-tuned and uploaded to my hugging face portfolio, with the help of Pradip videos")

st.image("toxic_img.png")
user_input = st.text_area('Enter the text to analyze')
button = st.button("Analyze")

d = {
    1: 'This is a toxic statement',
    0: 'This is a non-toxic statement'
}

if user_input and button:
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512, return_tensors='pt')
    #test-sample
    
    output = model(**test_sample)
    st.write("Logits: ", output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(), axis=1)
    styl = f"""
    <style>
    background-color: blue;
    color: white;
    </style>
    
    """
    prediction = "Prediction: This is a ", d[y_pred[0]], "statement"
    st.success( d[y_pred[0]])
    
  
