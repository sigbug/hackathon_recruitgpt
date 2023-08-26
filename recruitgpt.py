import openai
import PyPDF2
import docx
import os
import re
import pandas as pd
from pdfminer.high_level import extract_text
import json
import tempfile
import os
import streamlit as st

api_key = st.secrets.key
openai.api_key = api_key

def get_all_resumes(folder_path):
    resumes = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            text = convert_files_to_text(file_path)
            resumes.append(text)
    return resumes

def convert_files_to_text(file_path):
    try:
        if file_path.endswith('.pdf'):
            text = convert_pdf_to_text2(file_path)
        elif file_path.endswith('.docx'):
            text = convert_docx_to_text(file_path)
        elif file_path.endswith('.txt'):
            text = convert_txt_to_text(file_path)
        else:
            return ""
        return text
    except Exception as e:
        print(f"Error converting file {file_path} to text: {e}")
        return ""

def convert_pdf_to_text2(file_path):
    try:
        text = extract_text(file_path)
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {file_path}: {e}")
        return ""

def convert_pdf_to_text(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def convert_docx_to_text(file_path):
    try:
        doc = docx.Document(file_path)
        text = ''
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\\n'
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX {file_path}: {e}")
        return ""
    
def convert_txt_to_text(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except Exception as e:
        return f"An error occurred: {e}"

def get_choice_text_from_prompt(messages):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            temperature=0,
            max_tokens=4000
        )
        choice_text = response.choices[0]["message"]["content"]
        return choice_text
    except Exception as e:
        print("Error in get_choice_text_from_prompt:", str(e))
        return ""

def truncate_text_by_words(text, max_words=4000):
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
        resumetext = truncate_text_by_words(resumetext, 4000)

        system = """
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
          10. Soft skills
          11. Language proficiency
          12. Awards, Achievements and Accomplishments
          13. Adaptability and Learning Ability: Candidates who can quickly learn and adapt to new technologies, processes, or changes in the workplace.
          14. Extracurricular and Volunteer Activities: Involvement in community work or extracurricular activities that showcase leadership or teamwork skills.
          15. Professionalism and Formatting: A well-organized, error-free, and professional resume.
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
        answer should be in json format with keys as -
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
        """
        prompt = f"""
            Return details and score of the resume of candidate out of 100 against the given job description.
            Information about the candidate's resume and job description are given inside text delimited by triple backticks.

            Candidate's Resume :```{resumetext}```

            Job Description for the Target Role: ```{job_description}```
        """
        messages =  [
        {'role':'system', 'content':system},
        {'role':'user', 'content': prompt}]

        resume_detail = get_choice_text_from_prompt(messages)
        return resume_detail
    except Exception as e:
        print(f"Error analyzing resume: {e}")
        return ""

def analyze_all_resumes(folder_path, job_description):
  resumes = get_all_resumes(folder_path)
  scores = []
  for filename, resume_text in zip(os.listdir(folder_path), resumes):
      resume_detail = analyze_resume(resume_text, job_description)
      scores.append((filename, resume_detail))
  df = pd.DataFrame(scores, columns=["File_Name", "resume_detail"])
  return df

def extract_values(df):
    df['detail_json'] = df['resume_detail'].apply(json.loads)
    new_df = pd.DataFrame.from_records(df['detail_json'].tolist())
    new_df['File_Name'] = df['File_Name'].values
    new_df = new_df.sort_values(by='score', ascending=False)
    return new_df

# Define your functions here (convert_files_to_text, analyze_resume, extract_values)
def upload_file(files, job_description):

    file_paths = [file for file in files]

    result = []
    scores = []

    for file_path in file_paths:
        text = convert_files_to_text(file_path)
        result.append((file_path, text))

    for path, text in result:
        resume_score = analyze_resume(text, job_description)
        scores.append((path, resume_score))

    df = pd.DataFrame(scores, columns=["File_Name", "resume_detail"])
    new_df = extract_values(df)
    return new_df
# dasf
# Streamlit interface
st.title("RecruitGPT")

description = st.text_area("Job Description", height=400)
resumes = st.file_uploader("Resume", accept_multiple_files=True)

if st.button("Submit"):

    if resumes:
        temp_dir = tempfile.TemporaryDirectory()
        
        file_paths = []

        for file in resumes:
            with open(os.path.join(temp_dir.name, file.name), "wb") as f:
                f.write(file.read())
            file_paths.append(os.path.join(temp_dir.name, file.name))


        new_df = upload_file(file_paths, description)
        st.dataframe(new_df)
        # Save the new_df to a CSV file and provide a download link
        csv = new_df.to_csv(index=False)
        st.download_button("Download Result", data=csv, file_name='result.csv', mime='text/csv')
