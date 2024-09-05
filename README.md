[![Gnucoop Soc. Coop.](https://gnucoop.com/static/logo-gnu-71ab6373697553719b2d4ff79200204a.png)](https://gnucoop.com)


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
Groq, Together.ai, OpenAI, Ollama and BambooLLM

### Docker
docker build -t your-app-name

docker run -p 8501:8501 your-app-name
