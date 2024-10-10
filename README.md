The codebase is built on top of the excellent [RAGMEUP](https://github.com/AI-Commandos/RAGMeUp/tree/main).  
I currently added FastAPI and markdown support. There is more that needs to be added.  
Also this project will only support postgres
![API endpoints](./endpoints.png)

# Installation

## Server

```bash
git clone https://github.com/UnderstandLingBV/RAGMeUp.git
cd server
pip install -r requirements.txt
pip install --upgrade pip    
pip install "psycopg[binary,pool]"
