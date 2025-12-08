"""
MySQL Vector Database Handler

This module provides the MySQLVectorDB class for managing article embeddings
in MySQL using the native VECTOR data type. It mirrors the functionality of
the Pinecone implementation while using MySQL as the backend storage.
"""

import logging
from typing import List, Optional, Dict, Any
from sqlalchemy import text, and_, or_
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from align_data.db.session import make_session
from align_data.db.models import ArticleEmbedding, Article
from align_data.embeddings.mysql_vector.mysql_vector_models import MySQLVectorEntry

logger = logging.getLogger(__name__)


class MySQLVectorDB:
    """
    MySQL Vector Database Handler for article embeddings.
    
    This class provides methods to store, retrieve, and manage article embeddings
    in MySQL using the native VECTOR data type. It mirrors the functionality of
    the Pinecone implementation while using MySQL as the backend.
    """
    
    def __init__(self, session_factory=make_session):
        """
        Initialize the MySQL Vector Database handler.
        
        Args:
            session_factory: Factory function to create database sessions.
                           Defaults to make_session from align_data.db.session.
        """
        self.session_factory = session_factory
    
    def upsert_entry(self, entry: MySQLVectorEntry) -> None:
        """
        Insert or update embeddings for an article.
        
        This method will delete existing embeddings for the article and insert
        new ones, effectively performing an upsert operation.
        
        Args:
            entry: MySQLVectorEntry containing article hash ID and embeddings
            
        Raises:
            SQLAlchemyError: If database operation fails
        """
        if not entry.embeddings:
            logger.warning(f"No embeddings provided for article {entry.article_hash_id}")
            return
            
        try:
            with self.session_factory() as session:
                # First, delete existing embeddings for this article
                session.query(ArticleEmbedding).filter(
                    ArticleEmbedding.article_hash_id == entry.article_hash_id
                ).delete()
                
                # Convert embeddings to MySQL records
                mysql_records = entry.create_mysql_records()
                
                # Insert new embeddings
                for record in mysql_records:
                    embedding = ArticleEmbedding(
                        article_hash_id=record["article_hash_id"],
                        chunk_id=record["chunk_id"],
                        embedding_vector=str(record["embedding_vector"]),  # Convert to string for VectorType
                        text=record["text"]
                    )
                    session.add(embedding)
                
                session.commit()
                logger.info(f"Successfully upserted {len(mysql_records)} embeddings for article {entry.article_hash_id}")
                
        except IntegrityError as e:
            logger.error(f"Integrity error upserting embeddings for article {entry.article_hash_id}: {e}")
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error upserting embeddings for article {entry.article_hash_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error upserting embeddings for article {entry.article_hash_id}: {e}")
            raise
    
    def delete_entries(self, article_hash_ids: List[str]) -> None:
        """
        Delete all embeddings for given articles.
        
        Args:
            article_hash_ids: List of article hash IDs to delete embeddings for
            
        Raises:
            SQLAlchemyError: If database operation fails
        """
        if not article_hash_ids:
            logger.warning("No article hash IDs provided for deletion")
            return
            
        try:
            with self.session_factory() as session:
                deleted_count = session.query(ArticleEmbedding).filter(
                    ArticleEmbedding.article_hash_id.in_(article_hash_ids)
                ).delete(synchronize_session=False)
                
                session.commit()
                logger.info(f"Successfully deleted {deleted_count} embeddings for {len(article_hash_ids)} articles")
                
        except SQLAlchemyError as e:
            logger.error(f"Database error deleting embeddings for articles {article_hash_ids}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error deleting embeddings for articles {article_hash_ids}: {e}")
            raise
    
    def query_similar_vectors(
        self, 
        query_vector: List[float], 
        top_k: int = 10,
        exclude_hash_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find most similar vectors using MySQL vector search with article joins.
        
        Args:
            query_vector: The query vector to find similar embeddings for
            top_k: Number of top similar results to return
            exclude_hash_ids: List of article hash IDs to exclude from results
            
        Returns:
            List of dictionaries containing similarity results with metadata:
            - chunk_id: The chunk identifier
            - article_hash_id: The article hash ID
            - similarity_score: The similarity score (cosine similarity)
            - text: The chunk text
            - article_title: The article title
            - article_url: The article URL
            - article_source: The article source
            - article_authors: The article authors
            - date_published: The article publication date
            
        Raises:
            SQLAlchemyError: If database operation fails
        """
        if not query_vector:
            logger.warning("Empty query vector provided")
            return []
            
        exclude_hash_ids = exclude_hash_ids or []
        
        try:
            with self.session_factory() as session:
                # Convert query vector to string format for MySQL VECTOR operations
                query_vector_str = str(query_vector)
                
                # Build the base query with JOIN to articles table
                # Use MySQL's VECTOR_DISTANCE function for cosine similarity
                # Note: MySQL 8.0.34+ supports VECTOR data type and similarity functions
                similarity_expr = text(f"(1 - VECTOR_DISTANCE(embedding_vector, '{query_vector_str}', 'COSINE')) AS similarity_score")
                
                query = session.query(
                    ArticleEmbedding.chunk_id,
                    ArticleEmbedding.article_hash_id,
                    ArticleEmbedding.text,
                    Article.title.label('article_title'),
                    Article.url.label('article_url'),
                    Article.source.label('article_source'),
                    Article.authors.label('article_authors'),
                    Article.date_published,
                    similarity_expr
                ).join(
                    Article, ArticleEmbedding.article_hash_id == Article.id
                ).filter(
                    Article.is_valid == True
                )
                
                # Add exclusion filter if provided
                if exclude_hash_ids:
                    query = query.filter(
                        ArticleEmbedding.article_hash_id.notin_(exclude_hash_ids)
                    )
                
                # Order by similarity score (highest first)
                query = query.order_by(text("similarity_score DESC"))
                
                # Limit results
                query = query.limit(top_k)
                
                results = query.all()
                
                # Convert results to dictionary format
                similar_vectors = []
                for result in results:
                    similar_vectors.append({
                        'chunk_id': result.chunk_id,
                        'article_hash_id': result.article_hash_id,
                        'similarity_score': float(result.similarity_score),
                        'text': result.text,
                        'article_title': result.article_title,
                        'article_url': result.article_url,
                        'article_source': result.article_source,
                        'article_authors': result.article_authors,
                        'date_published': result.date_published
                    })
                
                logger.info(f"Found {len(similar_vectors)} similar vectors for query")
                return similar_vectors
                
        except SQLAlchemyError as e:
            logger.error(f"Database error querying similar vectors: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error querying similar vectors: {e}")
            raise
    
    def get_article_average_embedding(self, article_hash_id: str) -> Optional[List[float]]:
        """
        Calculate average embedding for an article by averaging all its chunk embeddings.
        
        Args:
            article_hash_id: The article hash ID to calculate average embedding for
            
        Returns:
            List of floats representing the average embedding, or None if no embeddings found
            
        Raises:
            SQLAlchemyError: If database operation fails
        """
        if not article_hash_id:
            logger.warning("Empty article hash ID provided")
            return None
            
        try:
            with self.session_factory() as session:
                # Get all embeddings for the article
                embeddings = session.query(ArticleEmbedding.embedding_vector).filter(
                    ArticleEmbedding.article_hash_id == article_hash_id
                ).all()
                
                if not embeddings:
                    logger.info(f"No embeddings found for article {article_hash_id}")
                    return None
                
                # Convert string representations back to lists and calculate average
                vectors = []
                for embedding in embeddings:
                    try:
                        # Parse the vector string back to list
                        vector = self._parse_vector_string(embedding.embedding_vector)
                        if vector is not None:
                            vectors.append(vector)
                    except Exception as e:
                        logger.warning(f"Failed to parse embedding vector for article {article_hash_id}: {e}")
                        continue
                
                if not vectors:
                    logger.warning(f"No valid embedding vectors found for article {article_hash_id}")
                    return None
                
                # Calculate average
                if len(vectors) == 1:
                    average_embedding = vectors[0]
                else:
                    # Calculate element-wise average
                    vector_length = len(vectors[0])
                    # Validate all vectors have the same length
                    if not all(len(v) == vector_length for v in vectors):
                        logger.error(f"Inconsistent vector dimensions for article {article_hash_id}")
                        return None
                    
                    average_embedding = []
                    for i in range(vector_length):
                        avg_value = sum(vector[i] for vector in vectors) / len(vectors)
                        average_embedding.append(avg_value)
                
                logger.info(f"Calculated average embedding for article {article_hash_id} from {len(vectors)} chunks")
                return average_embedding
                
        except SQLAlchemyError as e:
            logger.error(f"Database error calculating average embedding for article {article_hash_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calculating average embedding for article {article_hash_id}: {e}")
            raise
    
    def _parse_vector_string(self, vector_str: str) -> Optional[List[float]]:
        """
        Parse a vector string representation back to a list of floats.
        
        Args:
            vector_str: String representation of the vector
            
        Returns:
            List of floats, or None if parsing fails
        """
        if not vector_str:
            return None
            
        try:
            # Handle different possible formats
            if isinstance(vector_str, str):
                # Try to parse as Python literal (list format)
                import ast
                try:
                    parsed = ast.literal_eval(vector_str)
                    if isinstance(parsed, list) and all(isinstance(x, (int, float)) for x in parsed):
                        return [float(x) for x in parsed]
                except (ValueError, SyntaxError):
                    pass
                
                # Try to parse as JSON array
                import json
                try:
                    parsed = json.loads(vector_str)
                    if isinstance(parsed, list) and all(isinstance(x, (int, float)) for x in parsed):
                        return [float(x) for x in parsed]
                except (ValueError, json.JSONDecodeError):
                    pass
                    
            # If it's already a list, validate and convert
            elif isinstance(vector_str, list):
                if all(isinstance(x, (int, float)) for x in vector_str):
                    return [float(x) for x in vector_str]
                    
            return None
            
        except Exception:
            return None