from typing import List, Dict, Optional, Any, Tuple
import uuid
from datetime import datetime
from .llm_controller import LLMController
from .retrievers import ChromaRetriever
import json
import logging

logger = logging.getLogger(__name__)

class MemoryNote:
    """A memory note that represents a single unit of information in the memory system.
    
    This class encapsulates all metadata associated with a memory, including:
    - Core content and identifiers
    - Temporal information (creation and access times)
    - Semantic metadata (keywords, context, tags)
    - Relationship data (links to other memories)
    - Usage statistics (retrieval count)
    - Evolution tracking (history of changes)
    """
    
    def __init__(self, 
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None):
        """Initialize a new memory note with its associated metadata.
        
        Args:
            content (str): The main text content of the memory
            id (Optional[str]): Unique identifier for the memory. If None, a UUID will be generated
            keywords (Optional[List[str]]): Key terms extracted from the content
            links (Optional[Dict]): References to related memories
            retrieval_count (Optional[int]): Number of times this memory has been accessed
            timestamp (Optional[str]): Creation time in format YYYYMMDDHHMM
            last_accessed (Optional[str]): Last access time in format YYYYMMDDHHMM
            context (Optional[str]): The broader context or domain of the memory
            evolution_history (Optional[List]): Record of how the memory has evolved
            category (Optional[str]): Classification category
            tags (Optional[List[str]]): Additional classification tags
        """
        # Core content and ID
        self.content = content
        self.id = id or str(uuid.uuid4())
        
        # Semantic metadata
        self.keywords = keywords or []
        self.links = links or []
        self.context = context or "General"
        self.category = category or "Uncategorized"
        self.tags = tags or []
        
        # Temporal information
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        
        # Usage and evolution data
        self.retrieval_count = retrieval_count or 0
        self.evolution_history = evolution_history or []

class AgenticMemorySystem:
    """Core memory system that manages memory notes and their evolution.
    
    This system provides:
    - Memory creation, retrieval, update, and deletion
    - Content analysis and metadata extraction
    - Memory evolution and relationship management
    - Hybrid search capabilities
    """
    
    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 evolution_k: int = 5):
        """Initialize the memory system.

        Args:
            model_name: Name of the sentence transformer model
            llm_backend: LLM backend to use (openai/ollama)
            llm_model: Name of the LLM model
            evo_threshold: Number of memories before triggering evolution
            api_key: API key for the LLM service
            evolution_k: Number of nearest neighbors for link generation and evolution (paper default: 5)
        """
        self.memories = {}
        self.model_name = model_name
        self.evolution_k = evolution_k
        # Initialize ChromaDB retriever with empty collection
        try:
            # First try to reset the collection if it exists
            temp_retriever = ChromaRetriever(collection_name="memories",model_name=self.model_name)
            temp_retriever.client.reset()
        except Exception as e:
            logger.warning(f"Could not reset ChromaDB collection: {e}")
            
        # Create a fresh retriever instance
        self.retriever = ChromaRetriever(collection_name="memories",model_name=self.model_name)
        
        # Initialize LLM controller
        self.llm_controller = LLMController(llm_backend, llm_model, api_key)
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold

        # Ps2: Link generation prompt (paper Section 3.2, Eq.6)
        self._link_generation_prompt = """You are an AI memory linking agent.
Analyze the new memory and its nearest neighbors to determine
which memories should be linked based on shared semantic relationships.

New memory:
  content: {content}
  context: {context}
  keywords: {keywords}

Nearest neighbor memories:
{nearest_neighbors_memories}

Determine which neighbor memories share meaningful semantic connections
with the new memory. Consider:
- Common themes or topics
- Causal relationships
- Complementary information
- Shared conceptual frameworks

Return JSON:
{{
  "should_link": true or false,
  "connections": ["neighbor_memory_id_or_index"],
  "link_reasons": ["reason for each connection"]
}}
"""

        # Ps3: Memory evolution prompt (paper Section 3.3, Eq.7)
        # Called individually for each neighbor memory.
        self._evolution_prompt = """You are an AI memory evolution agent.
Given a new memory and a specific existing memory, determine
if the existing memory should be updated based on the new information.

New memory:
  content: {new_content}
  context: {new_context}
  keywords: {new_keywords}

Other context memories:
{other_neighbors}

Memory to evaluate for evolution:
  id: {target_id}
  content: {target_content}
  context: {target_context}
  keywords: {target_keywords}
  tags: {target_tags}

Rules:
- Set should_evolve=false if the new memory and target memory have no direct
  semantic relationship (e.g., personal info vs. unrelated technical topic).
- CRITICAL: Keep metadata focused on the TARGET memory's original content.
  Do NOT import unrelated concepts from the new memory.
- new_keywords: Max 7. Must all be relevant to the target memory's content.
  Keep the target's existing core keywords; only add/replace if directly relevant.
- new_context: Max 2 sentences, max 100 characters. Refine the target's context
  to reflect the new relationship, but keep it about the target's topic.
- new_tags: Max 5. Prune irrelevant tags. Each tag should classify the target memory.

Return JSON:
{{
  "should_evolve": true or false,
  "new_context": "updated context description",
  "new_tags": ["updated", "tags"],
  "new_keywords": ["updated", "keywords"]
}}
"""
        
    def analyze_content(self, content: str) -> Dict:            
        """Analyze content using LLM to extract semantic metadata.
        
        Uses a language model to understand the content and extract:
        - Keywords: Important terms and concepts
        - Context: Overall domain or theme
        - Tags: Classification categories
        
        Args:
            content (str): The text content to analyze
            
        Returns:
            Dict: Contains extracted metadata with keys:
                - keywords: List[str]
                - context: str
                - tags: List[str]
        """
        prompt = """Analyze the following content and extract structured metadata.
Follow the Zettelkasten principle: each note is an atomic, self-contained unit.

Rules:
- keywords: Extract 3 to 5 keywords that appear in or directly relate to the content.
  Only core concepts. Order from most to least important. No meta-descriptions.
- context: Exactly one sentence (max 80 characters) summarizing the main topic.
- tags: 2 to 3 broad category labels for classification. No overlapping or redundant tags.

Content for analysis:
""" + content
        try:
            response = self.llm_controller.llm.get_completion(prompt, response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "keywords": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "context": {
                                    "type": "string"
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                }
                            },
                            "required": ["keywords", "context", "tags"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }})
            result = json.loads(response)
            # Enforce caps even if LLM ignores maxItems
            if "keywords" in result:
                result["keywords"] = result["keywords"][:5]
            if "tags" in result:
                result["tags"] = result["tags"][:3]
            if "context" in result and len(result["context"]) > 120:
                result["context"] = result["context"][:120]
            return result
        except Exception as e:
            print(f"Error analyzing content: {e}")
            return {"keywords": [], "context": "General", "tags": []}

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note"""
        # Create MemoryNote without llm_controller
        if time is not None:
            kwargs['timestamp'] = time
        note = MemoryNote(content=content, **kwargs)

        # Populate keywords/context/tags via LLM analysis if not already provided
        if not note.keywords and not kwargs.get('keywords'):
            try:
                analysis = self.analyze_content(content)
                if analysis.get("keywords") and not note.keywords:
                    note.keywords = analysis["keywords"]
                if analysis.get("context") and note.context in ("General", "", None):
                    note.context = analysis["context"]
                if analysis.get("tags") and not note.tags:
                    existing = set(note.tags)
                    for tag in analysis["tags"]:
                        if tag not in existing:
                            note.tags.append(tag)
            except Exception as e:
                logger.warning("analyze_content failed, proceeding without enrichment: %s", e)

        # Process memory for link generation and evolution
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note

        # Add to ChromaDB with enriched document for embedding (paper Eq.3)
        metadata = {
            "id": note.id,
            "content": note.content,
            "keywords": note.keywords,
            "links": note.links,
            "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "context": note.context,
            "evolution_history": note.evolution_history,
            "category": note.category,
            "tags": note.tags
        }
        enriched_doc = self._build_enriched_document(note)
        self.retriever.add_document(enriched_doc, metadata, note.id)
        
        if evo_label == True:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id
    
    def consolidate_memories(self):
        """Consolidate memories: rebuild ChromaDB collection from in-memory state."""
        self.retriever = ChromaRetriever(collection_name="memories", model_name=self.model_name)

        for memory in self.memories.values():
            metadata = {
                "id": memory.id,
                "content": memory.content,
                "keywords": memory.keywords,
                "links": memory.links,
                "retrieval_count": memory.retrieval_count,
                "timestamp": memory.timestamp,
                "last_accessed": memory.last_accessed,
                "context": memory.context,
                "evolution_history": memory.evolution_history,
                "category": memory.category,
                "tags": memory.tags
            }
            enriched_doc = self._build_enriched_document(memory)
            self.retriever.add_document(enriched_doc, metadata, memory.id)
    
    def find_related_memories(self, query: str, k: int = 5) -> Tuple[str, List[int], List[str]]:
        """Find related memories using ChromaDB retrieval.

        Returns:
            Tuple of (formatted_text, indices, neighbor_ids)
            - formatted_text: human-readable string of neighbor memories
            - indices: position indices (0, 1, 2, ...)
            - neighbor_ids: actual ChromaDB document IDs for each result
        """
        if not self.memories:
            return "", [], []

        try:
            results = self.retriever.search(query, k)

            memory_str = ""
            indices = []
            neighbor_ids = []

            if 'ids' in results and results['ids'] and len(results['ids']) > 0 and len(results['ids'][0]) > 0:
                for i, doc_id in enumerate(results['ids'][0]):
                    if i < len(results['metadatas'][0]):
                        metadata = results['metadatas'][0][i]
                        memory_str += (
                            f"memory index:{i}\tid:{doc_id}\t"
                            f"talk start time:{metadata.get('timestamp', '')}\t"
                            f"memory content: {metadata.get('content', '')}\t"
                            f"memory context: {metadata.get('context', '')}\t"
                            f"memory keywords: {str(metadata.get('keywords', []))}\t"
                            f"memory tags: {str(metadata.get('tags', []))}\n"
                        )
                        indices.append(i)
                        neighbor_ids.append(doc_id)

            return memory_str, indices, neighbor_ids
        except Exception as e:
            logger.error(f"Error in find_related_memories: {str(e)}")
            return "", [], []

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """Find related memories using ChromaDB retrieval in raw format"""
        if not self.memories:
            return ""
            
        # Get results from ChromaDB
        results = self.retriever.search(query, k)
        
        # Convert to list of memories
        memory_str = ""
        
        if 'ids' in results and results['ids'] and len(results['ids']) > 0:
            for i, doc_id in enumerate(results['ids'][0][:k]):
                if i < len(results['metadatas'][0]):
                    # Get metadata from ChromaDB results
                    metadata = results['metadatas'][0][i]
                    
                    # Add main memory info
                    memory_str += f"talk start time:{metadata.get('timestamp', '')}\tmemory content: {metadata.get('content', '')}\tmemory context: {metadata.get('context', '')}\tmemory keywords: {str(metadata.get('keywords', []))}\tmemory tags: {str(metadata.get('tags', []))}\n"
                    
                    # Add linked memories if available
                    links = metadata.get('links', [])
                    j = 0
                    for link_id in links:
                        if link_id in self.memories and j < k:
                            neighbor = self.memories[link_id]
                            memory_str += f"talk start time:{neighbor.timestamp}\tmemory content: {neighbor.content}\tmemory context: {neighbor.context}\tmemory keywords: {str(neighbor.keywords)}\tmemory tags: {str(neighbor.tags)}\n"
                            j += 1
                            
        return memory_str

    def read(self, memory_id: str) -> Optional[MemoryNote]:
        """Retrieve a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to retrieve
            
        Returns:
            MemoryNote if found, None otherwise
        """
        return self.memories.get(memory_id)
    
    def update(self, memory_id: str, **kwargs) -> bool:
        """Update a memory note.
        
        Args:
            memory_id: ID of memory to update
            **kwargs: Fields to update
            
        Returns:
            bool: True if update successful
        """
        if memory_id not in self.memories:
            return False
            
        note = self.memories[memory_id]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(note, key):
                setattr(note, key, value)
                
        # Update in ChromaDB
        metadata = {
            "id": note.id,
            "content": note.content,
            "keywords": note.keywords,
            "links": note.links,
            "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "context": note.context,
            "evolution_history": note.evolution_history,
            "category": note.category,
            "tags": note.tags
        }
        
        # Delete and re-add to update (using enriched document for embedding)
        self.retriever.delete_document(memory_id)
        enriched_doc = self._build_enriched_document(note)
        self.retriever.add_document(document=enriched_doc, metadata=metadata, doc_id=memory_id)
        
        return True
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to delete
            
        Returns:
            bool: True if memory was deleted, False if not found
        """
        if memory_id in self.memories:
            # Delete from ChromaDB
            self.retriever.delete_document(memory_id)
            # Delete from local storage
            del self.memories[memory_id]
            return True
        return False
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using a hybrid retrieval approach."""
        # Get results from ChromaDB (only do this once)
        search_results = self.retriever.search(query, k)
        memories = []
        
        # Process ChromaDB results
        for i, doc_id in enumerate(search_results['ids'][0]):
            memory = self.memories.get(doc_id)
            if memory:
                memories.append({
                    'id': doc_id,
                    'content': memory.content,
                    'context': memory.context,
                    'keywords': memory.keywords,
                    'score': search_results['distances'][0][i]
                })
        
        return memories[:k]
    
    def search_agentic(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using ChromaDB retrieval."""
        if not self.memories:
            return []
            
        try:
            # Get results from ChromaDB
            results = self.retriever.search(query, k)
            
            # Process results
            memories = []
            seen_ids = set()
            
            # Check if we have valid results
            if ('ids' not in results or not results['ids'] or 
                len(results['ids']) == 0 or len(results['ids'][0]) == 0):
                return []
                
            # Process ChromaDB results
            for i, doc_id in enumerate(results['ids'][0][:k]):
                if doc_id in seen_ids:
                    continue
                    
                if i < len(results['metadatas'][0]):
                    metadata = results['metadatas'][0][i]
                    
                    # Create result dictionary with all metadata fields
                    memory_dict = {
                        'id': doc_id,
                        'content': metadata.get('content', ''),
                        'context': metadata.get('context', ''),
                        'keywords': metadata.get('keywords', []),
                        'tags': metadata.get('tags', []),
                        'timestamp': metadata.get('timestamp', ''),
                        'category': metadata.get('category', 'Uncategorized'),
                        'is_neighbor': False
                    }
                    
                    # Add score if available
                    if 'distances' in results and len(results['distances']) > 0 and i < len(results['distances'][0]):
                        memory_dict['score'] = results['distances'][0][i]
                        
                    memories.append(memory_dict)
                    seen_ids.add(doc_id)
            
            # Add linked memories (neighbors)
            neighbor_count = 0
            for memory in list(memories):  # Use a copy to avoid modification during iteration
                if neighbor_count >= k:
                    break

                # Get links from metadata
                links = memory.get('links', [])
                if not links and 'id' in memory:
                    # Try to get links from memory object
                    mem_obj = self.memories.get(memory['id'])
                    if mem_obj:
                        links = mem_obj.links

                for link_id in links:
                    if link_id not in seen_ids and link_id != memory.get('id') and neighbor_count < k:
                        neighbor = self.memories.get(link_id)
                        if neighbor:
                            memories.append({
                                'id': link_id,
                                'content': neighbor.content,
                                'context': neighbor.context,
                                'keywords': neighbor.keywords,
                                'tags': neighbor.tags,
                                'timestamp': neighbor.timestamp,
                                'category': neighbor.category,
                                'is_neighbor': True
                            })
                            seen_ids.add(link_id)
                            neighbor_count += 1

            # Increment retrieval_count and persist to ChromaDB
            current_time = datetime.now().strftime("%Y%m%d%H%M")
            for mem_dict in memories:
                mem_id = mem_dict.get('id')
                if mem_id and mem_id in self.memories:
                    note = self.memories[mem_id]
                    note.retrieval_count = (note.retrieval_count or 0) + 1
                    note.last_accessed = current_time
                    try:
                        updated_metadata = {
                            "id": note.id,
                            "content": note.content,
                            "keywords": note.keywords,
                            "links": note.links,
                            "retrieval_count": note.retrieval_count,
                            "timestamp": note.timestamp,
                            "last_accessed": note.last_accessed,
                            "context": note.context,
                            "evolution_history": note.evolution_history,
                            "category": note.category,
                            "tags": note.tags,
                        }
                        self.retriever.delete_document(mem_id)
                        enriched_doc = self._build_enriched_document(note)
                        self.retriever.add_document(enriched_doc, updated_metadata, mem_id)
                    except Exception as e:
                        logger.warning("Failed to update retrieval_count in ChromaDB for %s: %s", mem_id, e)

            return memories[:k]
        except Exception as e:
            logger.error(f"Error in search_agentic: {str(e)}")
            return []

    def _resolve_link_reference(
        self,
        reference: str,
        notes_id_list: list,
        search_indices: list,
    ) -> Optional[str]:
        """Resolve a link reference from LLM output to an actual memory UUID.

        The LLM may return references in various formats:
          - "0", "1" (plain index)
          - "memory index:0", "memory_index:0" (prefixed index)
          - An actual UUID string

        Returns the resolved UUID or None if unresolvable.
        """
        import re
        # Case 1: Already a valid UUID in our memories
        if reference in self.memories:
            return reference
        # Case 2: Extract numeric index from various formats
        match = re.search(r'(\d+)', reference.strip())
        if match:
            idx = int(match.group(1))
            # The index refers to the position in the search results
            if idx < len(search_indices):
                real_idx = search_indices[idx]
                if real_idx < len(notes_id_list):
                    return notes_id_list[real_idx]
            # Fallback: treat as direct index into notes_id_list
            if idx < len(notes_id_list):
                return notes_id_list[idx]
        return None

    def _format_neighbors_for_prompt(self, neighbor_ids: List[str]) -> str:
        """Format a list of neighbor memory IDs into a text block for LLM prompts."""
        text = ""
        for i, nid in enumerate(neighbor_ids):
            note = self.memories.get(nid)
            if note:
                text += (
                    f"memory index:{i}\tid:{nid}\t"
                    f"content: {note.content}\t"
                    f"context: {note.context}\t"
                    f"keywords: {note.keywords}\t"
                    f"tags: {note.tags}\n"
                )
        return text or "(none)"

    @staticmethod
    def _build_enriched_document(note: MemoryNote) -> str:
        """Build an enriched document string for embedding (paper Eq.3).

        Concatenates content, keywords, context, and tags so the embedding
        vector captures all semantic facets of the memory, not just content.
        """
        parts = [note.content]
        if note.keywords:
            parts.append(" ".join(note.keywords))
        if note.context and note.context != "General":
            parts.append(note.context)
        if note.tags:
            parts.append(" ".join(note.tags))
        return " | ".join(parts)

    def _persist_memory_to_chroma(self, note: MemoryNote):
        """Persist the current state of a memory note to ChromaDB.

        Uses delete-then-add pattern. If delete fails for a non-existence
        reason, the error is logged but we still attempt the add.
        """
        metadata = {
            "id": note.id,
            "content": note.content,
            "keywords": note.keywords,
            "links": note.links,
            "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "context": note.context,
            "evolution_history": note.evolution_history,
            "category": note.category,
            "tags": note.tags,
        }
        try:
            self.retriever.delete_document(note.id)
        except ValueError:
            # Document does not exist yet -- expected on first persist
            pass
        except Exception as e:
            logger.warning("Failed to delete document %s before re-add: %s", note.id, e)
        enriched_doc = self._build_enriched_document(note)
        self.retriever.add_document(enriched_doc, metadata, note.id)

    def _generate_links(self, note: MemoryNote, neighbors_text: str, neighbor_ids: List[str]) -> bool:
        """Phase 1 (Ps2): Link generation between new memory and its neighbors.

        Determines which neighbor memories should be linked to the new memory
        based on shared semantic relationships. Creates bidirectional links.

        Returns True if any links were created.
        """
        prompt = self._link_generation_prompt.format(
            content=note.content,
            context=note.context,
            keywords=note.keywords,
            nearest_neighbors_memories=neighbors_text,
        )

        try:
            response = self.llm_controller.llm.get_completion(
                prompt,
                response_format={"type": "json_schema", "json_schema": {
                    "name": "link_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "should_link": {"type": "boolean"},
                            "connections": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "link_reasons": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                        },
                        "required": ["should_link", "connections", "link_reasons"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }},
            )
            response_json = json.loads(response)

            if not response_json.get("should_link", False):
                return False

            connections = response_json.get("connections", [])
            indices = list(range(len(neighbor_ids)))

            resolved = []
            for conn in connections:
                r = self._resolve_link_reference(conn, neighbor_ids, indices)
                if r and r != note.id:
                    resolved.append(r)

            if not resolved:
                return False

            # Bidirectional linking
            for linked_id in resolved:
                # Forward: note -> linked
                if linked_id not in note.links:
                    note.links.append(linked_id)
                # Reverse: linked -> note
                linked_note = self.memories.get(linked_id)
                if linked_note and note.id not in linked_note.links:
                    linked_note.links.append(note.id)
                    self._persist_memory_to_chroma(linked_note)

            note.links = list(dict.fromkeys(note.links))  # deduplicate
            logger.info(
                "Link generation: note %s linked to %s",
                note.id, resolved,
            )
            return True

        except json.JSONDecodeError as e:
            logger.error("Malformed JSON from LLM in link generation: %s", e)
            return False
        except KeyError as e:
            logger.error("Missing key in link generation response: %s", e)
            return False

    def _evolve_neighbors(self, note: MemoryNote, neighbor_ids: List[str]) -> bool:
        """Phase 2 (Ps3): Per-neighbor memory evolution.

        For each neighbor, individually query the LLM to decide whether
        the neighbor should be updated given the new memory.
        Paper: Eq.7 -- each m_j is evaluated separately.

        Returns True if at least one neighbor was evolved.
        """
        any_evolved = False

        for target_id in neighbor_ids:
            target = self.memories.get(target_id)
            if not target:
                continue

            other_ids = [nid for nid in neighbor_ids if nid != target_id]
            other_text = self._format_neighbors_for_prompt(other_ids)

            prompt = self._evolution_prompt.format(
                new_content=note.content,
                new_context=note.context,
                new_keywords=note.keywords,
                other_neighbors=other_text,
                target_id=target_id,
                target_content=target.content,
                target_context=target.context,
                target_keywords=target.keywords,
                target_tags=target.tags,
            )

            try:
                response = self.llm_controller.llm.get_completion(
                    prompt,
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "evolution_response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_evolve": {"type": "boolean"},
                                "new_context": {"type": "string"},
                                "new_tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "new_keywords": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["should_evolve", "new_context", "new_tags", "new_keywords"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    }},
                )
                response_json = json.loads(response)

                if response_json.get("should_evolve", False):
                    # Apply with caps to prevent metadata bloat
                    new_ctx = response_json["new_context"]
                    target.context = new_ctx[:150] if len(new_ctx) > 150 else new_ctx
                    target.tags = response_json["new_tags"][:5]
                    new_kw = response_json.get("new_keywords", target.keywords)
                    target.keywords = new_kw[:7]
                    target.evolution_history.append({
                        "timestamp": datetime.now().strftime("%Y%m%d%H%M"),
                        "trigger": note.id,
                        "action": "evolution",
                    })
                    self._persist_memory_to_chroma(target)
                    any_evolved = True
                    logger.info(
                        "Evolution: neighbor %s evolved (trigger=%s)",
                        target_id, note.id,
                    )

            except json.JSONDecodeError as e:
                logger.error("Malformed JSON from LLM for neighbor %s: %s", target_id, e)
                continue
            except KeyError as e:
                logger.error("Missing key in evolution response for neighbor %s: %s", target_id, e)
                continue

        return any_evolved

    def process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        """Process a memory note: link generation (Ps2) then per-neighbor evolution (Ps3).

        Paper-aligned two-phase processing:
        - Phase 1 (Eq.6): Link generation -- determine which neighbors should be linked
        - Phase 2 (Eq.7): Memory evolution -- update each neighbor individually

        Args:
            note: The memory note to process

        Returns:
            Tuple[bool, MemoryNote]: (did_evolve, processed_note)
        """
        if not self.memories:
            return False, note

        try:
            neighbors_text, indices, neighbor_ids = self.find_related_memories(
                note.content, k=self.evolution_k
            )
            if not neighbors_text or not neighbor_ids:
                return False, note

            # Phase 1: Link Generation (Ps2, Eq.6)
            linked = self._generate_links(note, neighbors_text, neighbor_ids)

            # Phase 2: Memory Evolution (Ps3, Eq.7) -- per-neighbor
            evolved = self._evolve_neighbors(note, neighbor_ids)

            return (linked or evolved), note

        except Exception as e:
            logger.error("Error in process_memory: %s", e)
            return False, note
