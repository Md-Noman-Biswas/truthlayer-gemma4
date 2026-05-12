import os
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

class DocumentStore:
    def __init__(self, db_path="chroma_db", collection_name="medical_docs"):
        # Initialize persistent ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # Load embedding model (fast, local)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        
    def _chunk_text(self, text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
        """Simple character-based chunking."""
        chunks = []
        i = 0
        while i < len(text):
            chunk = text[i:i + chunk_size]
            if len(chunk.strip()) > 50: # ignore tiny chunks
                chunks.append(chunk.strip())
            i += chunk_size - overlap
        return chunks

    def ingest_pdf(self, file_path: str, source_name: str) -> int:
        """Parse PDF, chunk text, and store embeddings in ChromaDB."""
        reader = PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
                
        chunks = self._chunk_text(full_text)
        if not chunks:
            return 0
            
        embeddings = self.encoder.encode(chunks).tolist()
        
        # Generate unique IDs
        ids = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source_name} for _ in chunks]
        
        self.collection.upsert(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieve most relevant chunks for a query."""
        if self.collection.count() == 0:
            return []
            
        query_embedding = self.encoder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count())
        )
        
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metadatas = results["metadatas"][0]
            # Format nicely with source
            formatted_chunks = [f"[Source: {meta['source']}]\n{doc}" for doc, meta in zip(docs, metadatas)]
            return formatted_chunks
        return []
        
# Global singleton instance for easy import
document_store = DocumentStore()
