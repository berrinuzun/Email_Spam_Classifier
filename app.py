import pickle
import streamlit as st

def load_model():
    with open('model', 'rb') as f:
        return pickle.load(f)

def load_vectorizer():
    with open('vectorizer', 'rb') as f:
        return pickle.load(f)

def main():
    
    st.title("Email Spam Classifier")
    st.markdown("With this application, you can accurately classify emails as spam or not spam.")
    st.divider()
    
    email_text = st.text_area("Enter email text", "")
    
    if st.button("Classify"):
        
        if email_text:
            
            with st.spinner("Classifying..."):

                vectorizer = load_vectorizer()

                email_text_transformed = vectorizer.transform([email_text])  
                
                model = load_model()

                result = model.predict(email_text_transformed)
            
            st.write(f"Prediction: {'Spam' if result == 1 else 'Not Spam'}")
        else:
            st.error("Please enter the email text.")

if __name__ == '__main__':
    main()
