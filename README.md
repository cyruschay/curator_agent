"Download Ollama at https://ollama.com/download"

ollama pull bge-large:latest
ollama pull gpt-oss:20b

conda create --name gly_env python=3.12
"type yes to install"
conda activate gly_env

cd <curation_agent directory>
pip install -r requirements.txt

bash main_pc.sh
