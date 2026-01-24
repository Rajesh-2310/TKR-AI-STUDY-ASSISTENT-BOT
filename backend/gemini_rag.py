"""
Gemini-powered RAG Engine for TKR Chatbot
Uses Google's Gemini AI with retrieval augmented generation
"""
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import time
from database import get_db
import logging
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Force reload .env to avoid caching issues
load_dotenv(override=True)

logger = logging.getLogger(__name__)

class GeminiRAGEngine:
    """RAG Engine using Gemini AI for answer generation"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', gemini_api_key=None):
        """Initialize RAG engine with embedding model and Gemini"""
        try:
            # Load embedding model for semantic search
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
            
            # Configure Gemini - force reload env vars
            load_dotenv(override=True)
            api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")
            
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel(os.getenv('GEMINI_MODEL', 'gemini-pro'))
            logger.info("Initialized Gemini AI model")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini RAG engine: {e}")
            raise
    
    def generate_embedding(self, text):
        """Generate embedding vector for text"""
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def store_embeddings(self, material_id, chunks):
        """Store document chunks and their embeddings"""
        try:
            db = get_db()
            embeddings_data = []
            
            for idx, chunk in enumerate(chunks):
                embedding = self.generate_embedding(chunk['text'])
                embeddings_data.append((
                    material_id,
                    chunk['text'],
                    idx,
                    chunk.get('page', 0),
                    json.dumps(embedding)
                ))
            
            query = """
                INSERT INTO document_embeddings 
                (material_id, chunk_text, chunk_index, page_number, embedding_vector)
                VALUES (%s, %s, %s, %s, %s)
            """
            
            db.execute_many(query, embeddings_data)
            logger.info(f"Stored {len(embeddings_data)} embeddings for material {material_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            raise
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def search_similar_chunks(self, query, subject_id=None, top_k=5):
        """Search for most similar document chunks with optimized vectorization"""
        try:
            db = get_db()
            
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            query_vec = np.array(query_embedding)
            
            # Optimized query: limit results and use indexes
            # Only fetch top 200 candidates to reduce data transfer
            if subject_id:
                sql = """
                    SELECT de.id, de.chunk_text, de.page_number, 
                           de.material_id, de.embedding_vector,
                           m.title, m.subject_id
                    FROM document_embeddings de
                    JOIN materials m ON de.material_id = m.id
                    WHERE m.subject_id = %s AND m.is_processed = TRUE
                    ORDER BY de.id DESC
                    LIMIT 200
                """
                embeddings = db.execute_query(sql, (subject_id,))
            else:
                sql = """
                    SELECT de.id, de.chunk_text, de.page_number, 
                           de.material_id, de.embedding_vector,
                           m.title, m.subject_id
                    FROM document_embeddings de
                    JOIN materials m ON de.material_id = m.id
                    WHERE m.is_processed = TRUE
                    ORDER BY de.id DESC
                    LIMIT 200
                """
                embeddings = db.execute_query(sql)
            
            if not embeddings:
                logger.warning("No embeddings found in database")
                return []
            
            # Vectorized similarity calculation
            embedding_vectors = []
            metadata = []
            
            for emb in embeddings:
                try:
                    stored_embedding = json.loads(emb['embedding_vector'])
                    if stored_embedding and len(stored_embedding) > 0:
                        embedding_vectors.append(stored_embedding)
                        metadata.append({
                            'chunk_text': emb['chunk_text'],
                            'page_number': emb['page_number'],
                            'material_id': emb['material_id'],
                            'material_title': emb['title']
                        })
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping invalid embedding: {e}")
                    continue
            
            if not embedding_vectors:
                logger.warning("No valid embeddings to search")
                return []
            
            # Convert to NumPy array for vectorized operations
            embedding_matrix = np.array(embedding_vectors)
            
            # Vectorized cosine similarity calculation (much faster than loop)
            # similarity = (A · B) / (||A|| * ||B||)
            dot_products = np.dot(embedding_matrix, query_vec)
            query_norm = np.linalg.norm(query_vec)
            embedding_norms = np.linalg.norm(embedding_matrix, axis=1)
            similarities = dot_products / (embedding_norms * query_norm)
            
            # Get top k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Build results
            results = []
            for idx in top_indices:
                result = metadata[idx].copy()
                result['similarity'] = float(similarities[idx])
                results.append(result)
            
            logger.info(f"Found {len(results)} similar chunks (searched {len(embeddings)} candidates)")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def generate_answer_with_gemini(self, query, context_chunks):
        """Generate answer using Gemini AI with retrieved context"""
        try:
            if not context_chunks:
                # No context found - alert about missing content
                prompt = f"""You are a Senior Engineering Professor at TKR College.

Student's Question: {query}

⚠️ IMPORTANT: No relevant content found in uploaded study materials.

RESPONSE FORMAT:
## ⚠️ Topic Not Found in Your Materials

[Explain what's missing and suggest uploading relevant materials]

### 🤔 Would you like standard reference material?

[If providing answer, use this structure:]
## 📖 Definition
## 🔍 Explanation  
## 💡 Key Points
## 📐 Formulas (LaTeX: $$formula$$)

Use markdown, emojis, and clear formatting:"""
                
                response = self.gemini_model.generate_content(prompt)
                return {
                    'answer': response.text,
                    'sources': [],
                    'confidence': 0.3
                }
            
            # Combine context from retrieved chunks with source citations
            context_parts = []
            for chunk in context_chunks:
                source_ref = f"[Ref: {chunk['material_title']}, Page {chunk['page_number']}]"
                context_parts.append(f"{source_ref}:\n{chunk['chunk_text']}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Calculate average confidence
            avg_confidence = sum(chunk['similarity'] for chunk in context_chunks) / len(context_chunks)
            
            # Create optimized prompt for fast, conversational answers
            prompt = f"""You are an AI tutor for TKR College students. Answer directly and conversationally using the study materials provided.

STUDY MATERIALS:
{context}

STUDENT'S QUESTION: {query}

INSTRUCTIONS:
• Answer directly - don't say "According to the material..." just explain naturally
• Use markdown formatting with ## headers and emojis (📖, 🔍, 💡, ⚡, ✅)
• **Bold** key terms, `code style` for technical terms
• Use bullet points and numbered lists
• Include LaTeX for formulas: $$formula$$
• ALWAYS cite sources at the end (MANDATORY)

STRUCTURE:
## 📖 [Main Topic]
[Direct explanation]

## 🔍 Key Points
• **Point 1**: Explanation
• **Point 2**: Explanation

## 📐 Formulas (if applicable)
$$formula$$

---
## 📚 Sources
**Referenced from:** [Material names and pages]

Generate a clear, well-formatted answer:"""
            
            # Generate response with Gemini
            response = self.gemini_model.generate_content(prompt)
            
            # Extract sources
            sources = []
            for chunk in context_chunks:
                source_info = {
                    'material': chunk['material_title'],
                    'page': chunk['page_number'],
                    'material_id': chunk['material_id']
                }
                if source_info not in sources:
                    sources.append(source_info)
            
            return {
                'answer': response.text,
                'sources': sources,
                'confidence': float(avg_confidence),
                'context_chunks': context_chunks
            }
            
        except Exception as e:
            logger.error(f"Gemini answer generation failed: {e}")
            return {
                'answer': f"I encountered an error while generating the answer: {str(e)}",
                'sources': [],
                'confidence': 0.0
            }
    
    def answer_question(self, question, subject_id=None, top_k=5):
        """Complete RAG pipeline: retrieve and generate answer with Gemini (with caching)"""
        try:
            # Create cache key
            cache_key = f"{question.lower().strip()}_{subject_id}_{top_k}"
            
            # Check cache first
            if hasattr(self, '_answer_cache') and cache_key in self._answer_cache:
                cached_result, cache_time = self._answer_cache[cache_key]
                # Cache valid for 5 minutes
                if (time.time() - cache_time) < 300:
                    logger.info(f"Returning cached answer for: {question[:50]}")
                    return cached_result
            
            # Search for relevant chunks
            similar_chunks = self.search_similar_chunks(question, subject_id, top_k)
            
            # Generate answer using Gemini
            result = self.generate_answer_with_gemini(question, similar_chunks)
            
            # Cache the result
            if not hasattr(self, '_answer_cache'):
                self._answer_cache = {}
            
            # Limit cache size to 100 entries
            if len(self._answer_cache) > 100:
                # Remove oldest entries
                oldest_keys = sorted(self._answer_cache.keys(), 
                                   key=lambda k: self._answer_cache[k][1])[:20]
                for key in oldest_keys:
                    del self._answer_cache[key]
            
            self._answer_cache[cache_key] = (result, time.time())
            
            return result
            
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return {
                'answer': "An error occurred while processing your question. Please try again.",
                'sources': [],
                'confidence': 0.0
            }

# Global Gemini RAG engine instance
gemini_rag_engine = None

def get_gemini_rag_engine():
    """Get or create Gemini RAG engine instance"""
    global gemini_rag_engine
    if gemini_rag_engine is None:
        gemini_rag_engine = GeminiRAGEngine()
    return gemini_rag_engine
