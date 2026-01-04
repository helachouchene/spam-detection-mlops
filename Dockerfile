FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt requirements_streamlit.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements_streamlit.txt
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
COPY . .
EXPOSE 5000 8501
CMD ["python", "api/app.py"]