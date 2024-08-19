# dinopandasai

### CREATE a python environment
python -m venv env

### ACTIVATE ENV

. env/bin/activate

### INSTALL requirements
pip install -r requirements

### RUN APP
streamlit run app.py

### LLM PROVIDER SUPPORTED 
Groq, Together.ai, OpenAI and BambooLLM

### Docker
docker build -t your-app-name

docker run -p 8501:8501 your-app-name
