#importing libraries
import streamlit as st
import helper
import pickle

#importing the model
model = pickle.load(open("forestmodel.pkl", "rb"))

#title
st.header("Check Duplicate Questions Pairs")


#define two text input widgets to get input for question 1 and question 2
q1 = st.text_input("Enter the Question 1")
q2 = st.text_input("Enter the Question 2")

#checking if the "Find" button is clicked
if st.button("Find"):
    #creating a query point using the input questions
    query = helper.query_point_creator(q1, q2)

    #making a prediction using the query point
    result = model.predict(query)[0]

    #checing the result and display whether the questions are duplicates or not
    if result:
        st.header("These are Duplicate Questions Pairs.")
    else:
        st.header("These are not Duplicate Questions Pairs.")