from langchain_community.embeddings import JinaEmbeddings, OllamaEmbeddings, HuggingFaceEmbeddings
import os
import dotenv


def get_embedding_function():
    dotenv.load_dotenv()

    # Set up embedding handling for vector store
    if os.getenv('force_cpu') == "True":
        model_kwargs = {
            'device': 'cpu'
        }
    else:
        model_kwargs = {
            'device': 'cuda'
        }

    if os.getenv('embedding_provider') == 'jina':
        embeddings = JinaEmbeddings(
            model_name=os.getenv('embedding_model'),
        )
    elif os.getenv('embedding_provider') == 'ollama':
        embeddings = OllamaEmbeddings(
            model_name=os.getenv('embedding_model'),
        )
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv('embedding_model'),
            model_kwargs=model_kwargs
        )

    return embeddings
