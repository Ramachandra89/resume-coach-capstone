import streamlit as st
from resume_parser import ResumeParser
from llm_service import LLMService
import tempfile
import os

def main():
    st.title("AI Resume Coach")
    st.write("Upload your resume and job description to get personalize coaching")

    #Initialize services
    resume_parser = ResumeParser()
    llm_service = LLMService()

    #File Upload
    uploaded_resume = st.file_uploader("Upload your resume", type=['pdf','txt'])

    #Job Description Input
    job_description = st.text_area("Paste the job description here:", height=200)

    if uploaded_resume and job_description:
        if st.button("Analyze Resume"):
            with st.spinner("Analyzing your resume..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_resume.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_resume.getvalue())
                    tmp_file_path = tmp_file.name

                try:
                    #Parse resume
                    resume_text = resume_parser.parse_resume(tmp_file_path)

                    #Analyze resume
                    analysis = llm_service.analyze_resume(resume_text, job_description)

                    # Display results
                    st.subheader("Resume Analysis")
                    st.write(analysis)

                except Exception as e:
                    st.error(f"Error processing your resume: {str(e)}")
                finally:
                    #Clean up temporary file
                    os.unlink(tmp_file_path)

    #Chat Interface
    st.subheader("Chat with Resume Coach")
    user_message = st.text_input("Ask your question")

    if user_message:
        if st.button("Send"):
            with st.spinner("Getting response..."):
                response = llm_service.chat_with_coach(user_message)
                st.write(response)

if __name__ == "__main__":
    main()