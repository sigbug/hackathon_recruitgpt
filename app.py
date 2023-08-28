import pandas as pd
import docx
import os
import re
from pdfminer.high_level import extract_text
import json
import tempfile
import streamlit as st
from langchain.llms import Clarifai
import urllib

#set the PAT
CLARIFAI_PAT = st.secrets.key

llm = Clarifai(pat=CLARIFAI_PAT, user_id='clarifai', app_id='ml', model_id='llama2-70b-chat-alternative')
template = """
Temperature: {temp}
top_k: {top_k}
top_p: {top_p}
"""

### Read the resume-
def convert_files_to_text(file_path):
    try:
        if file_path.endswith('.pdf'):
            text = convert_pdf_to_text(file_path)
        elif file_path.endswith('.docx'):
            text = convert_docx_to_text(file_path)
        elif file_path.endswith('.txt'):
            text = convert_to_text(file_path)
        else:
            return ""
        return text
    except Exception as e:
        print(f"Error converting file {file_path} to text: {e}")
        return ""

def convert_pdf_to_text(file_path):
    try:
        text = extract_text(file_path)
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {file_path}: {e}")
        return ""

def convert_docx_to_text(file_path):
    try:
        doc = docx.Document(file_path)
        text = ''
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n'
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX {file_path}: {e}")
        return ""
    
def convert_to_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return f"Error: The file '{file_path}' was not found."
    except UnicodeDecodeError:
        return "Error: The file could not be decoded using UTF-8 encoding."
    except Exception as e:
        return f"Error: {str(e)}"


###
def truncate_text_by_words(text, max_words=1000):
    """
    Truncates the text to a specified number of words.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])



def analyze_resume(resumetext,job_description):
    try:
        # Ensure resume text does not exceed 4000 tokens
        resumetext = truncate_text_by_words(resumetext, 1000)

        resume_score_prompt = f"""
        <s>
        [INST]
        <<SYS>>
        You are an excellent talent recuiter and your task is to score resumes of candidates between 0-100 against a job description.
        You will be provided with candidate resume and job description.

        The system instruction is:
        Step-1: First check whether the candidate's resume is an actual resume or not.
        Step-2: If the candidate's resume is not an actual resume then score=0, else further
        analyse the candidate's resume against the job description by looking for these following qualities:
          1. Relevant Experience: Relevant work experience in the field or industry related to the job role
          2. Duration of experiences
          3. Previous job titles
          4. Specific responsibilities and their impact
          5. Achievements in previous experiences
          6. Education - The candidate's educational background
          7. Educational quality
          8. Certifications: specialized training, especially if they align with the job requirements.
          9. Technical skills
        Step-3:
        Score the overall quality of resume against the job description and skills required between 0-100.
        Scoring resume for the job should be as detailed as possible taking into account all the qualities.
        Score should be such that it can be compared against different candidate's resumes for shortlisting purpose.
        Score should be a floating point number with upto 2 decimal point accuracy.
        Step-4:
        Only return-
        the final score of resume,
        name,
        gmail,
        list of social media links,
        list of skillset and expertise,
        list of relevant skillset and expertise,
        list of certifications,
        explanation of projects under 200 tokens,
        list of explanation of position of responsibilities under 200 tokens,
        years of experience,
        experience description under 200 tokens,
        relevant experience description under 200 tokens,
        list of educational qualification,
        detailed explanation of the scoring procedure under 200 tokes,
        list of extracurriculars under 200 words,
        list of awards and achievements,
        list of previous job title.
        Answer should be in json format with keys as -
        score,
        name,
        gmail,
        social media,
        skillset and expertise,
        relevant skillset and expertise,
        certifications,
        projects,
        position of responsibilities,
        years of experience,
        experience description,
        relevant experience description,
        educational qualification,
        score explanation,
        extracurricular,
        Awards and Achievements,
        previous job titles
        If a key value is missing in the resume then value should be null.
        <</SYS>>


        User:
        Return details and score of the resume of candidate out of 100 against the given job description.
        Information about the candidate's resume and job description are given inside text delimited by triple backticks.

        Candidate's Resume :```{resumetext}```

        Job Description for the Target Role: ```{job_description}```
        [/INST]
        """

        resume_detail = llm.predict(text=resume_score_prompt, temp=0, top_k=5, top_p=1.0)
        return resume_detail
    except Exception as e:
        print(f"Error analyzing resume: {e}")
        return ""



def prepare_questions(resumetext,job_description):
    try:
        # Ensure resume text does not exceed 4000 tokens
        resumetext = truncate_text_by_words(resumetext,1000)
        prepare_question_prompt = f"""
        <s>
        [INST]
        <<SYS>>
        You are an experienced talent recruiter with a proven track record of selecting the best candidates for job roles.
        Your current task is to craft interview questions that thoroughly evaluate candidates' fitness for a specific job role based on their resumes and
        the provided job description. Your aim is to identify candidates who possess the essential skills and qualities required for the role.
        You will be provided with candidate resume and job description.

        The system instruction is:
        Step-1:
        Carefully analyze each candidate's resume in the context of the job description. Pay attention to the following aspects:
          1. Relevant Experience: Work experience that directly relates to the job role or the industry.
          2. Duration of Experiences: How long the candidate has been engaged in relevant roles.
          3. Previous Job Titles: Titles that align with the role's responsibilities and expectations.
          4. Specific Responsibilities and Impact: Instances where the candidate's contributions had a significant impact.
          5. Achievements in Previous Experiences: Noteworthy accomplishments that showcase the candidate's abilities.
          6. Education: The candidate's educational background and its relevance.
          7. Educational Quality: The reputation of the institutions the candidate attended.
          8. Certifications: Specialized training that enhances the candidate's qualifications.
          9. Technical Skills: Proficiency in relevant tools, technologies, and methodologies.
          10. Projects: Projects that directly relates to the job role.

        Step-2: Craft a set of the top 10 interview questions that delve into candidates' relevant experience, skills, and qualities.
        Your questions should uncover their strengths, weaknesses, and alignment with the role's requirements.
        For each question, specifically mention the project name or experience mentioned in the resume to provide context to the candidate.

        Step-3: Provide the list of 10 questions only in a JSON format, with each question labeled as Question 1, Question 2, and so on.
        <</SYS>>

        User:
        Kindly prepare a set of 10 comprehensive interview questions tailored to assess my suitability for the job role.
        Ensure that the questions are based on the experiences and projects mentioned in my resume and the provided job description.

        Information about the candidate's resume and job description are given inside text delimited by triple backticks.

        Candidate's Resume :```{resumetext}```
        Job Description for the Target Role: ```{job_description}```
        [/INST]
        """
        interview_questions = llm.predict(text=prepare_question_prompt, temp=0, top_k=5, top_p=1.0)
        return interview_questions
    except Exception as e:
        print(f"Error analyzing resume: {e}")
        return ""

## ... [rest of your imports and functions remain unchanged] ...

def streamlit_get_scores(file_paths, job_description):
    result = []
    scores = []
    
    for file_path in file_paths:
        text = convert_files_to_text(file_path)
        result.append((file_path, text))

    for path, text in result:
        resume_detail = analyze_resume(text, job_description)
        try:
            detail_dict = json.loads(resume_detail)
            score = detail_dict.get('score', None)
        except json.JSONDecodeError:
            print("Error decoding the JSON from the resume analysis.")
            score = None
        scores.append((path, score))

    df = pd.DataFrame(scores, columns=["File_Name", "score"])
    return df


# Streamlit interface
st.title("Recruit-Llama")

# Textbox for job description
description = st.text_area("Job Description", height=500)

# Upload multiple resumes
resumes = st.file_uploader("Upload Resumes", accept_multiple_files=True)

if st.button("Submit"):
    if resumes:
        temp_dir = tempfile.TemporaryDirectory()
        file_paths = []
        for file in resumes:
            with open(os.path.join(temp_dir.name, file.name), "wb") as f:
                f.write(file.read())
            file_paths.append(os.path.join(temp_dir.name, file.name))

        # Get scores for all uploaded resumes
        scores_df = streamlit_get_scores(file_paths, description)

        # Display all resumes with their scores
        st.subheader("All Resumes with Scores")
        st.dataframe(scores_df)

        # Sort the dataframe based on score and take top 3
        top_3_resumes = scores_df.sort_values(by="score", ascending=False).head(3)

        # Prepare questions for top 3 resumes and display clickable links
        st.subheader("Top 3 Resumes with Questions")
        for index, row in top_3_resumes.iterrows():
            text = convert_files_to_text(row['File_Name'])
            questions = prepare_questions(text, description)
            encoded_questions = urllib.parse.quote(questions)
            
            # Create a link to a secondary Streamlit app with the encoded questions as a query parameter
            link = f"http://localhost:8502/?questions={encoded_questions}"
            st.markdown(f"[{row['File_Name']} - Score: {row['score']}]({link})")
