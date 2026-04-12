import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ..config import settings
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    _instance = None
    _vectorstore = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize ChromaDB with HuggingFace Embeddings"""
        if not settings.enable_vector_db:
            logger.info("Vector DB is disabled in settings.")
            return

        try:
            # Ensure data directory exists
            os.makedirs(os.path.dirname(settings.chroma_db_path), exist_ok=True)
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
            )
            
            self._vectorstore = Chroma(
                collection_name="yta_topics",
                embedding_function=self.embeddings,
                persist_directory=settings.chroma_db_path
            )
            logger.info(f"Vector store initialized at {settings.chroma_db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            self._vectorstore = None

    def add_topic(self, topic_id: str, topic_text: str, metadata: dict = None):
        """Add a new topic to the vector store"""
        if not self._vectorstore:
            return None
        
        try:
            self._vectorstore.add_texts(
                texts=[topic_text],
                metadatas=[metadata or {}],
                ids=[topic_id]
            )
            return topic_id
        except Exception as e:
            logger.error(f"Failed to add topic to vector store: {str(e)}")
            return None

    def find_similar_topics(self, query_text: str, k: int = 5):
        """Find semantically similar topics"""
        if not self._vectorstore:
            return []
        
        try:
            results = self._vectorstore.similarity_search_with_score(query_text, k=k)
            # Normalize scores (Chroma scores are L2 distance, lower is more similar)
            # We want to return a list of {topic, similarity_score}
            similar_topics = []
            for doc, distance in results:
                # Approximate similarity (1 / (1 + distance))
                similarity = 1.0 / (1.0 + distance)
                similar_topics.append({
                    "topic": doc.page_content,
                    "similarity": similarity,
                    "metadata": doc.metadata
                })
            return similar_topics
        except Exception as e:
            logger.error(f"Failed to find similar topics: {str(e)}")
            return []

# Singleton instance
vector_store = VectorStore()
