import replicate
import os
from dotenv import load_dotenv
import re

class LLMService:
    def __init__(self):
        # Load .env file from the current directory
        dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(dotenv_path)

        # Get token
        self.api_token = os.getenv('REPLICATE_API_TOKEN')
        if not self.api_token:
            raise ValueError("REPLICATE_API_TOKEN not found in environment variables.")

        # Set the token explicitly for Replicate
        os.environ["REPLICATE_API_TOKEN"] = self.api_token

    def _clean_text(self, text):
        """Clean and normalize text input."""
        if not text:
            return ""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

    def _chunk_text(self, text, max_chars=1000):
        """Split text into smaller chunks."""
        if not text:
            return []
        
        # Clean the text first
        text = self._clean_text(text)
        
        # If text is short enough, return as is
        if len(text) <= max_chars:
            return [text]
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > max_chars and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def analyze_resume(self, resume_text, job_description):
        try:
            # Validate inputs
            if not resume_text or not job_description:
                return "Error: Resume text and job description are required."

            # Clean and chunk the inputs
            resume_chunks = self._chunk_text(resume_text)
            job_desc_chunks = self._chunk_text(job_description)

            # Use the first chunk of each for initial analysis
            # This helps avoid token limit issues
            resume_chunk = resume_chunks[0]
            job_desc_chunk = job_desc_chunks[0]

            prompt = f"""
            Analyze the following resume against the job description and provide detailed feedback:
            
            Resume:
            {resume_chunk}
            
            Job Description:
            {job_desc_chunk}
            
            #Instructions:
            1. Act as a recruiter, review the resume and highlight weak areas, overused buzzwords, and missing metrics.
            2. Suggest changes to sound more results driven, quantifiable and compelling to fit the job description provided
            3. Update the resume to be fully optimized for Applicant Tracking Systems (ATS)
            4. Use industry specific keywords. Write a powerful, 3-line professional summary that hooks a recruiter in under ten seconds.
            5. Prioritize impact, clarity and value. Rephrase the experience section to highlight impact, results, and transferable skills using action verbs and quantifiable outcomes.
            6. Suggest a clean, modern resume format that works for both humans and ATS. No graphics, no columns. Just structured and effective.
            7. Tailor the resume to fit the specific job description provided. Highlight matching experience and reword sections to match the language used.
            8. Write a compelling cover letter based on the resume and job description. Keep it personal, enthusiastic and under 200 words.
            9. Act as a hiring manager. Based on the provided job description, what would a resume of a top candidate look like? Compare it to the one provided and suggest what needs to be changed.
            """

            try:
                output = replicate.run(
                    "replicate/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
                    input={
                        "prompt": prompt,
                        "system_prompt": "You are a helpful, accurate, and professional resume coach. Provide clear, direct, and actionable feedback",
                        "temperature": 0.7,
                        "max_length": 2000,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1
                    }
                )
                
                if isinstance(output, list):
                    return "".join(output)
                return output

            except Exception as e:
                # If the first attempt fails, try with a smaller model
                output = replicate.run(
                    "replicate/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d",
                    input={
                        "prompt": prompt,
                        "system_prompt": "You are a helpful, accurate, and professional resume coach. Provide clear, direct, and actionable feedback",
                        "temperature": 0.7,
                        "max_length": 2000,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1
                    }
                )
                
                if isinstance(output, list):
                    return "".join(output)
                return output

        except Exception as e:
            return f"Error analyzing resume: {str(e)}"

    def chat_with_coach(self, user_message, context=None):
        try:
            # Clean the input
            user_message = self._clean_text(user_message)
            context = self._clean_text(context) if context else None

            few_shot_examples = """
            You are a professional resume coach. Be concise, supportive, and specific.

            #Instructions:
            1. Act as a recruiter, review the resume and highlight weak areas, overused buzzwords, and missing metrics.
            2. Suggest changes to sound more results driven, quantifiable and compelling to fit the job description provided
            3. Update the resume to be fully optimized for Applicant Tracking Systems (ATS)
            4. Use industry specific keywords. Write a powerful, 3-line professional summary that hooks a recruiter in under ten seconds.
            5. Prioritize impact, clarity and value. Rephrase the experience section to highlight impact, results, and transferable skills using action verbs and quantifiable outcomes.
            6. Suggest a clean, modern resume format that works for both humans and ATS. No graphics, no columns. Just structured and effective.
            7. Tailor the resume to fit the specific job description provided. Highlight matching experience and reword sections to match the language used.
            8. Write a compelling cover letter based on the resume and job description. Keep it personal, enthusiastic and under 200 words.
            9. Act as a hiring manager. Based on the provided job description, what would a resume of a top candidate look like? Compare it to the one provided and suggest what needs to be changed.
            """

            user_prompt = f"""
            Previous context: {context if context else 'None'}

            Now here's a new case:
            User: {user_message}
            Coach:
            """

            full_prompt = few_shot_examples + "\n" + user_prompt

            try:
                output = replicate.run(
                    "replicate/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
                    input={
                        "prompt": full_prompt,
                        "temperature": 0.7,
                        "max_length": 2000,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1
                    }
                )

                if isinstance(output, list):
                    return "".join(output)
                return output

            except Exception as e:
                # If the first attempt fails, try with a smaller model
                output = replicate.run(
                    "replicate/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d",
                    input={
                        "prompt": full_prompt,
                        "temperature": 0.7,
                        "max_length": 2000,
                        "top_p": 0.9,
                        "repetition_penalty": 1.1
                    }
                )

                if isinstance(output, list):
                    return "".join(output)
                return output

        except Exception as e:
            return f"Error in chat: {str(e)}"

