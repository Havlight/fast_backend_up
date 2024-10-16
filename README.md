The codebase is built on top of the excellent [RAGMEUP](https://github.com/AI-Commandos/RAGMeUp/tree/main).  
I currently added FastAPI and markdown support. There is more that needs to be added.  
Also this project will only support postgres

![API endpoints](./endpoints.png)

# Installation

## Server

```bash
git clone https://github.com/Havlight/fast_backend_up.git
cd server
pip install -r requirements.txt
pip install --upgrade pip    
pip install "psycopg[binary,pool]"
```
Then run the server using `python main.py` from the server subfolder.
## configuration
you can add your api key in `.env`
```bash
GOOGLE_API_KEY=put_your_key-fqxkE4Y
JINA_API_KEY=put_your_key
```
specify embedding model
```bash
embedding_model=jina-embeddings-v2-base-zh
embedding_provider=jina
```

add jwt key
```bash
jwt_key=your_key
```

use docker
```bash
# seperate docker
#vector_store_uri='postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
#vector_store_sparse_uri='postgresql://langchain:langchain@localhost:6024/langchain'
# docker-compose
vector_store_uri='postgresql+psycopg://langchain:langchain@db:5432/langchain'
vector_store_sparse_uri='postgresql://langchain:langchain@db:5432/langchain'
```
