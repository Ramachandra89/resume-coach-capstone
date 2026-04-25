import PyPDF2
from docx import Document
import os

class ResumeParser:
    def __init__(self):
        self.supported_formats = ['.pdf']

    def parse_resume(self, file_path):
        """
        Parse resume from different file formats
        """
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format. Supported formats are {self.supported_formats}")
        
        if file_extension == '.pdf':
            return self._parse_pdf(file_path)
        elif file_extension == 'docx':
            return self._parse_docx(file_path)
        else:
            return self._parse_txt(file_path)
        
    def _parse_pdf(self, file_path):
        """Parse PDF Resume"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            raise Exception(f"Error parsing PDF file: {str(e)}")

        