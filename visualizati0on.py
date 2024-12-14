import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import pickle
import pandas as pd
st.title("Customer Churn Prediction Model")
st.header("Boost Your Business")
with open(r"C:\Users\anjil\OneDrive\Desktop\churn_model\churn_data.pkl" , "rb") as file:
    model=pickle.load(file)
st.sidebar.header("Customer Churn")
#st.write(pd.read_csv(r"C:\Users\anjil\OneDrive\Desktop\churn_model\customer\customer_churn_dataset-training-master.csv"))
age=st.sidebar.number_input("Age" , 7 , 90)
gender=st.sidebar.selectbox("Gender" ,["Male" ,"Female" , "Others"])
tenure=st.sidebar.slider("Tenure" , 0 , 100)
usage_frequency=st.sidebar.slider("Usage_frequency" , 0, 100)
support_calls=st.sidebar.number_input("Support Calls" , 0 , 50)
payment_delay=st.sidebar.number_input("Payment Delay" , 0 , 50)
subscription_type=st.sidebar.selectbox("Subscription Type" ,["Premium" , "Standard" , "Basic"] )
contract_length= st.sidebar.selectbox("Contract_Length" , ["Annual" ,"Monthly" , "Quarterly"])
total_spend=st.sidebar.number_input("Total_spend" , value=0 )
last_interaction=st.sidebar.slider("Last Interaction" , 0, 100)
st.subheader("Upload your CSV file")

uploaded_file =st.file_uploader("Choose a CSV file" , type="csv")
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    df=df.iloc[: , 1:12]
    prediction = model.predict(df)
    if st.button("Download ouput CSV"):    # Convert the DataFrame to a CSV in memory
        buffer = io.StringIO()
        mf=pd.DataFrame()
        mf["Ouput"]=prediction
        mf.to_csv(buffer, index=False)
        
        buffer.seek(0)  # Rewind the buffer to the beginning
        # Create a download button for the CSV file
        st.download_button(
            label="Download CSV",
            data=buffer.getvalue(),
            file_name="output.csv",
            mime="text/csv"
            )
st.title("For Individual Data")
st.header("Indiviual Data Result from Side Bar")
variables=[age , gender , tenure , usage_frequency , support_calls, payment_delay , subscription_type , contract_length , total_spend ,last_interaction ]
numpy_arra = np.array(variables)
#numpy_array=pd.DataFrame(numpy_arra , columns=["Age","Gender","Tenure","Usage Frequency","Support Calls","Payment Delay","Subscription Type","Contract Length","Total Spend","Last Interaction"])
if st.button("predict"):
    predicted=model.predict(numpy_arra.reshape(1, 10))
    #st.subheader("Result ::  " , predicted)
    st.subheader(f"Result :: {str(predicted)}")

st.markdown("---")  # A separator line

st.subheader("About the Creator")
st.markdown("""
    This Customer Churn Prediction Model was created by [Anjil Timsina](anjiltimsina1234@gmail.com).
    If you have any questions, feel free to contact me at:
    - Email: [anjiltimsina1234@gmail.com]
    - LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/in/anjiltimsina/)
    - GitHub: [Your GitHub Profile](https://github.com/)anjiltimsina)

    Thanks for using the model!
""")