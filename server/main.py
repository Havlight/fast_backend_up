from fastapi.responses import FileResponse
from RAGHelper_cloud import RAGHelperCloud
from RAGHelper_local import RAGHelperLocal
from fastapi import FastAPI, HTTPException, Request, Depends

from db import User, create_db_and_tables
from schemas import UserCreate, UserRead, UserUpdate
from users import auth_backend, current_active_user, fastapi_users
from contextlib import asynccontextmanager

from pydantic import BaseModel
import logging
import os
from dotenv import load_dotenv
import uvicorn
from typing import List, Optional, Dict, Any

# add user endpoints
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Not needed if you setup a migration system like Alembic
    await create_db_and_tables()
    yield

# Initialize FastAPI application
app = FastAPI(lifespan=lifespan)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.include_router(
    fastapi_users.get_auth_router(auth_backend), prefix="/auth/jwt", tags=["auth"]
)
app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_reset_password_router(),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_verify_router(UserRead),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)


@app.get("/authenticated-route")
async def authenticated_route(user: User = Depends(current_active_user)):
    return {"message": f"Hello {user.email}!"}

# Disable parallelism in tokenizers to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()

# Instantiate the RAG Helper class based on the environment configuration
if any(os.getenv(key) == "True" for key in ["use_openai", "use_gemini", "use_azure", "use_ollama"]):
    logger.info("Instantiating the cloud RAG helper.")
    raghelper = RAGHelperCloud(logger)
else:
    logger.info("Instantiating the local RAG helper.")
    raghelper = RAGHelperLocal(logger)


class Document(BaseModel):
    filename: str


@app.post("/add_local_document",tags=['RAG'])
async def add_document(doc: Document,user: User = Depends(current_active_user)):
    """
    Add a document to the RAG helper.

    This endpoint expects a JSON payload containing the filename of the document to be added.
    It then invokes the addDocument method of the RAG helper to store the document.

    Returns:
        JSON response with the filename.
    """
    filename = doc.filename

    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    file_types = ["md", "pdf", "txt", "csv", "docx", "pptx", "xml", "json"]
    if not any(filename.endswith(ext) for ext in file_types):
        raise HTTPException(status_code=400, detail="invalid filetype")

    logger.info(f"Adding document {filename}")
    raghelper.addDocument(filename)

    return {"filename": filename}


class ChatRequest(BaseModel):
    prompt: str
    history: list = []
    docs: list = []


# Response models
class DocumentResponse(BaseModel):
    s: str  # source
    c: str  # content
    pk: Optional[str] = None  # primary key (if present)
    provenance: Optional[float] = None  # provenance score (if present)


class ChatResponse(BaseModel):
    reply: str
    history: List[Dict[str, str]]
    documents: List[DocumentResponse]
    rewritten: bool
    question: str


@app.post("/chat", response_model=ChatResponse,tags=['RAG'])
async def chat(request: ChatRequest,user: User = Depends(current_active_user)):
    """
    Handle chat interactions with the RAG system.

    This endpoint processes the user's prompt, retrieves relevant documents,
    and returns the assistant's reply along with conversation history.

    Returns:
        JSON response containing the assistant's reply, history, documents, and other metadata.
    """
    prompt = request.prompt
    history = request.history
    original_docs = request.docs
    docs = original_docs

    # Get the LLM response
    (new_history, response) = raghelper.handle_user_interaction(prompt, history)
    if not docs or 'docs' in response:
        docs = response['docs']

    # Break up the response for local LLMs
    if isinstance(raghelper, RAGHelperLocal):
        end_string = os.getenv("llm_assistant_token")
        reply = response['text'][response['text'].rindex(end_string) + len(end_string):]

        # Get updated history
        new_history = [{"role": msg["role"], "content": msg["content"].format_map(response)} for msg in new_history]
        new_history.append({"role": "assistant", "content": reply})
    else:
        # Populate history for other LLMs
        new_history = [{"role": msg[0], "content": msg[1].format_map(response)} for msg in new_history]
        new_history.append({"role": "assistant", "content": response['answer']})
        reply = response['answer']

    # Format documents
    if not original_docs or 'docs' in response:
        new_docs = [{
            's': doc.metadata['source'],
            'c': doc.page_content,
            **({'pk': doc.metadata['pk']} if 'pk' in doc.metadata else {}),
            **({'provenance': float(doc.metadata['provenance'])} if 'provenance' in doc.metadata else {})
        } for doc in docs if 'source' in doc.metadata]
    else:
        new_docs = docs

    # Build the response dictionary
    response_dict = {
        "reply": reply,
        "history": new_history,
        "documents": new_docs,
        "rewritten": False,
        "question": prompt
    }

    # Check for rewritten question
    if os.getenv("use_rewrite_loop") == "True" and prompt != response['question']:
        response_dict["rewritten"] = True
        response_dict["question"] = response['question']

    return response_dict


# Response model for list of document filenames
class DocumentsResponse(BaseModel):
    files: List[str]


@app.get("/get_documents", response_model=DocumentsResponse,tags=['RAG'])
async def get_documents(user: User = Depends(current_active_user)):
    """
    Retrieve a list of documents from the data directory.

    This endpoint checks the configured data directory and returns a list of files
    that match the specified file types.

    Returns:
        JSON response containing the list of files.
    """
    data_dir = os.getenv('data_directory')
    file_types = os.getenv("file_types", "").split(",")

    # Filter files based on specified types
    files = [f for f in os.listdir(data_dir)
             if os.path.isfile(os.path.join(data_dir, f)) and os.path.splitext(f)[1][1:] in file_types]

    return {"files": files}


class DocumentRequest(BaseModel):
    filename: str


@app.post("/get_document",tags=['RAG'])
async def get_document(doc_request: DocumentRequest,user: User = Depends(current_active_user)):
    """
    Retrieve a specific document from the data directory.

    This endpoint expects a JSON payload containing the filename of the document to retrieve.
    If the document exists, it is sent as a file response.

    Returns:
        JSON response with the error message and HTTP status code 404 if not found,
        otherwise sends the file as an attachment.
    """
    filename = doc_request.filename
    data_dir = os.getenv('data_directory')
    file_path = os.path.join(data_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path,
                        media_type='application/octet-stream',
                        filename=filename)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
