from setuptools import setup, find_packages

setup(
    name="MCQGEN",
    version="1.0.0",
    author="Pradeep Kumar Singha",
    author_email="mr.pradeepkumarsingha@gmail.com",
    description="A tool to generate multiple-choice questions.",
    install_requires=["openai","langchain","streamlit","python-dotenv","PyPDF2"],
    packages=find_packages(),
)