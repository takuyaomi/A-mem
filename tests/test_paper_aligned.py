"""Tests for paper-aligned link generation and evolution (arXiv:2502.12110).

Covers:
1. Link generation (Ps2) as a separate step
2. Bidirectional links
3. Per-neighbor evolution (Ps3)
4. Evolution persistence to ChromaDB
5. Index/ID mapping correctness
6. find_related_memories returns neighbor_ids
7. _persist_memory_to_chroma helper
8. _format_neighbors_for_prompt helper
9. evolution_k parameter
"""
import json
import os
import unittest
from datetime import datetime
from unittest.mock import patch, MagicMock

from agentic_memory.memory_system import AgenticMemorySystem, MemoryNote
from tests.test_utils import MockLLMController

# Ensure a dummy API key so OpenAIController.__init__ does not raise
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-unit-tests")


def _create_system_with_mock(evolution_k: int = 5) -> AgenticMemorySystem:
    """Create an AgenticMemorySystem with a MockLLMController injected."""
    system = AgenticMemorySystem(
        model_name="all-MiniLM-L6-v2",
        llm_backend="openai",
        llm_model="gpt-4o-mini",
        evolution_k=evolution_k,
    )
    mock = MockLLMController()
    system.llm_controller.llm = mock
    return system


def _add_memory_directly(system: AgenticMemorySystem, content: str,
                         keywords=None, context=None, tags=None,
                         note_id=None) -> str:
    """Add a memory directly (bypassing LLM analysis) for test setup."""
    note = MemoryNote(
        content=content,
        id=note_id,
        keywords=keywords or ["test"],
        context=context or "test context",
        tags=tags or ["test"],
    )
    system.memories[note.id] = note
    metadata = {
        "id": note.id, "content": note.content,
        "keywords": note.keywords, "links": note.links,
        "retrieval_count": note.retrieval_count,
        "timestamp": note.timestamp, "last_accessed": note.last_accessed,
        "context": note.context, "evolution_history": note.evolution_history,
        "category": note.category, "tags": note.tags,
    }
    system.retriever.add_document(note.content, metadata, note.id)
    return note.id


class TestFindRelatedMemoriesReturnsNeighborIds(unittest.TestCase):
    """Test that find_related_memories returns the 3-tuple including neighbor_ids."""

    def test_returns_three_values(self):
        system = _create_system_with_mock()
        id1 = _add_memory_directly(system, "Python programming")
        id2 = _add_memory_directly(system, "Python data science")

        text, indices, neighbor_ids = system.find_related_memories("Python", k=2)
        self.assertIsInstance(text, str)
        self.assertIsInstance(indices, list)
        self.assertIsInstance(neighbor_ids, list)
        self.assertEqual(len(indices), len(neighbor_ids))
        # All returned IDs should be valid memory IDs
        for nid in neighbor_ids:
            self.assertIn(nid, system.memories)

    def test_empty_memories_returns_empty(self):
        system = _create_system_with_mock()
        text, indices, neighbor_ids = system.find_related_memories("anything")
        self.assertEqual(text, "")
        self.assertEqual(indices, [])
        self.assertEqual(neighbor_ids, [])

    def test_neighbor_ids_contain_actual_doc_ids(self):
        """The neighbor_ids should be the actual ChromaDB document IDs, not indices."""
        system = _create_system_with_mock()
        id1 = _add_memory_directly(system, "Machine learning algorithms")
        id2 = _add_memory_directly(system, "Deep learning neural networks")
        id3 = _add_memory_directly(system, "Natural language processing")

        text, indices, neighbor_ids = system.find_related_memories("neural networks", k=3)
        self.assertTrue(len(neighbor_ids) > 0)
        for nid in neighbor_ids:
            self.assertTrue(
                len(nid) > 10,  # UUIDs are long strings
                f"Expected UUID-like ID, got: {nid}"
            )

    def test_formatted_text_includes_id_field(self):
        """The formatted text should now include the id field."""
        system = _create_system_with_mock()
        id1 = _add_memory_directly(system, "Test content")
        text, indices, neighbor_ids = system.find_related_memories("Test", k=1)
        self.assertIn("id:", text)


class TestLinkGeneration(unittest.TestCase):
    """Test Phase 1 (Ps2): Link generation as a separate step."""

    def test_links_created_when_should_link_true(self):
        system = _create_system_with_mock()
        id1 = _add_memory_directly(system, "Python programming basics")
        id2 = _add_memory_directly(system, "Python data structures")

        mock: MockLLMController = system.llm_controller.llm
        # Ps1: analyze content
        mock.analyze_response = json.dumps({
            "keywords": ["Python", "functions"],
            "context": "Python programming",
            "tags": ["programming"],
        })
        # Ps2: link generation -- link to both existing memories
        mock.link_response = json.dumps({
            "should_link": True,
            "connections": ["0", "1"],
            "link_reasons": ["same topic", "related topic"],
        })
        # Ps3: no evolution
        mock.evolution_responses = [
            json.dumps({"should_evolve": False, "new_context": "", "new_tags": [], "new_keywords": []}),
            json.dumps({"should_evolve": False, "new_context": "", "new_tags": [], "new_keywords": []}),
        ]

        id3 = system.add_note("Python functions and methods")
        note3 = system.read(id3)

        # The new note should have links to the existing memories
        self.assertTrue(len(note3.links) > 0, "New note should have links")

    def test_no_links_when_should_link_false(self):
        system = _create_system_with_mock()
        id1 = _add_memory_directly(system, "Quantum physics theory")

        mock: MockLLMController = system.llm_controller.llm
        mock.analyze_response = json.dumps({
            "keywords": ["cooking", "recipes"],
            "context": "Cooking",
            "tags": ["food"],
        })
        mock.link_response = json.dumps({
            "should_link": False,
            "connections": [],
            "link_reasons": [],
        })
        mock.evolution_responses = [
            json.dumps({"should_evolve": False, "new_context": "", "new_tags": [], "new_keywords": []}),
        ]

        id2 = system.add_note("Cooking Italian pasta recipes")
        note2 = system.read(id2)
        self.assertEqual(len(note2.links), 0, "Should have no links when should_link=False")

    def test_self_loop_prevention(self):
        """Links should not include the note's own ID."""
        system = _create_system_with_mock()
        id1 = _add_memory_directly(system, "Test content A")

        mock: MockLLMController = system.llm_controller.llm
        mock.analyze_response = json.dumps({
            "keywords": ["test"],
            "context": "Test",
            "tags": ["test"],
        })
        # LLM tries to link to index 0 (which is the only existing memory)
        mock.link_response = json.dumps({
            "should_link": True,
            "connections": ["0"],
            "link_reasons": ["related"],
        })
        mock.evolution_responses = [
            json.dumps({"should_evolve": False, "new_context": "", "new_tags": [], "new_keywords": []}),
        ]

        id2 = system.add_note("Test content B")
        note2 = system.read(id2)

        # Should link to id1, not to itself
        if note2.links:
            self.assertNotIn(id2, note2.links, "Note should not link to itself")

    def test_link_deduplication(self):
        """Duplicate links should be removed."""
        system = _create_system_with_mock()
        id1 = _add_memory_directly(system, "Topic A content")

        mock: MockLLMController = system.llm_controller.llm
        mock.analyze_response = json.dumps({
            "keywords": ["topic"],
            "context": "Topic",
            "tags": ["test"],
        })
        # LLM returns the same connection twice
        mock.link_response = json.dumps({
            "should_link": True,
            "connections": ["0", "0"],
            "link_reasons": ["same", "same"],
        })
        mock.evolution_responses = [
            json.dumps({"should_evolve": False, "new_context": "", "new_tags": [], "new_keywords": []}),
        ]

        id2 = system.add_note("Topic A related")
        note2 = system.read(id2)

        # Links should be deduplicated
        if note2.links:
            self.assertEqual(
                len(note2.links), len(set(note2.links)),
                "Links should not contain duplicates"
            )


class TestBidirectionalLinks(unittest.TestCase):
    """Test that links are created bidirectionally (A->B and B->A)."""

    def test_bidirectional_link_creation(self):
        system = _create_system_with_mock()
        id1 = _add_memory_directly(system, "Neural network architectures")
        id2 = _add_memory_directly(system, "Deep learning training methods")

        mock: MockLLMController = system.llm_controller.llm
        mock.analyze_response = json.dumps({
            "keywords": ["neural", "networks"],
            "context": "Deep learning",
            "tags": ["AI"],
        })
        # Link new note to both existing
        mock.link_response = json.dumps({
            "should_link": True,
            "connections": ["0", "1"],
            "link_reasons": ["related topic", "complementary"],
        })
        mock.evolution_responses = [
            json.dumps({"should_evolve": False, "new_context": "", "new_tags": [], "new_keywords": []}),
            json.dumps({"should_evolve": False, "new_context": "", "new_tags": [], "new_keywords": []}),
        ]

        id3 = system.add_note("Convolutional neural network layers")
        note3 = system.read(id3)

        # Forward links: note3 -> id1, note3 -> id2
        linked_ids = set()
        for link in note3.links:
            linked_ids.add(link)

        # Check that at least one of id1, id2 is in note3's links
        self.assertTrue(
            linked_ids.intersection({id1, id2}),
            f"note3 should link to id1 or id2. Links: {note3.links}"
        )

        # Reverse links: id1/id2 -> note3
        for target_id in linked_ids.intersection({id1, id2}):
            target = system.read(target_id)
            self.assertIn(
                id3, target.links,
                f"Memory {target_id} should have reverse link to {id3}"
            )

    def test_bidirectional_links_persist_in_chroma(self):
        """Reverse links should be persisted to ChromaDB."""
        system = _create_system_with_mock()
        id1 = _add_memory_directly(system, "Topic X details")

        mock: MockLLMController = system.llm_controller.llm
        mock.analyze_response = json.dumps({
            "keywords": ["topic"],
            "context": "Topic X",
            "tags": ["test"],
        })
        mock.link_response = json.dumps({
            "should_link": True,
            "connections": ["0"],
            "link_reasons": ["related"],
        })
        mock.evolution_responses = [
            json.dumps({"should_evolve": False, "new_context": "", "new_tags": [], "new_keywords": []}),
        ]

        id2 = system.add_note("Topic X advanced")

        # Verify ChromaDB has the updated reverse link for id1 via direct ID lookup
        results = system.retriever.collection.get(ids=[id1], include=["metadatas"])
        self.assertTrue(len(results['ids']) > 0, f"id1 should exist in ChromaDB")
        chroma_meta = results['metadatas'][0]
        chroma_links = chroma_meta.get('links', [])
        if isinstance(chroma_links, str):
            chroma_links = json.loads(chroma_links)
        self.assertIn(
            id2, chroma_links,
            f"ChromaDB metadata for {id1} should contain reverse link to {id2}"
        )


class TestPerNeighborEvolution(unittest.TestCase):
    """Test Phase 2 (Ps3): Each neighbor is evolved individually."""

    def test_individual_evolution_calls(self):
        """Each neighbor should get its own LLM call for evolution."""
        system = _create_system_with_mock()
        id1 = _add_memory_directly(system, "Machine learning basics",
                                    keywords=["ML"], tags=["AI"])
        id2 = _add_memory_directly(system, "Statistical learning theory",
                                    keywords=["statistics"], tags=["math"])
        id3 = _add_memory_directly(system, "Data preprocessing pipelines",
                                    keywords=["data"], tags=["engineering"])

        mock: MockLLMController = system.llm_controller.llm
        mock.analyze_response = json.dumps({
            "keywords": ["supervised", "learning"],
            "context": "Supervised ML",
            "tags": ["AI"],
        })
        mock.link_response = json.dumps({
            "should_link": True,
            "connections": ["0"],
            "link_reasons": ["related"],
        })
        # Each neighbor gets different evolution response
        mock.evolution_responses = [
            json.dumps({
                "should_evolve": True,
                "new_context": "ML basics including supervised learning",
                "new_tags": ["AI", "supervised"],
                "new_keywords": ["ML", "supervised"],
            }),
            json.dumps({
                "should_evolve": True,
                "new_context": "Statistical foundations of supervised methods",
                "new_tags": ["math", "supervised"],
                "new_keywords": ["statistics", "supervised"],
            }),
            json.dumps({
                "should_evolve": False,
                "new_context": "",
                "new_tags": [],
                "new_keywords": [],
            }),
        ]

        mock.reset_call_tracking()
        id4 = system.add_note("Supervised learning algorithms")

        # Count evolution-related calls (Ps3 calls)
        evolution_calls = [
            c for c in mock.call_history
            if "memory evolution agent" in c["prompt"]
        ]
        # Should have one call per neighbor (up to evolution_k)
        self.assertEqual(
            len(evolution_calls), 3,
            f"Expected 3 evolution calls (one per neighbor), got {len(evolution_calls)}"
        )

    def test_evolved_neighbors_have_updated_metadata(self):
        """Neighbors that are evolved should have updated context/tags/keywords."""
        system = _create_system_with_mock()
        id1 = _add_memory_directly(system, "Original content",
                                    keywords=["original"], context="Original context",
                                    tags=["original_tag"])

        mock: MockLLMController = system.llm_controller.llm
        mock.analyze_response = json.dumps({
            "keywords": ["new"],
            "context": "New context",
            "tags": ["new_tag"],
        })
        mock.link_response = json.dumps({
            "should_link": False,
            "connections": [],
            "link_reasons": [],
        })
        mock.evolution_responses = [
            json.dumps({
                "should_evolve": True,
                "new_context": "Evolved context reflecting new information",
                "new_tags": ["evolved_tag", "updated"],
                "new_keywords": ["evolved", "keyword"],
            }),
        ]

        id2 = system.add_note("New related content")

        note1 = system.read(id1)
        self.assertEqual(note1.context, "Evolved context reflecting new information")
        self.assertEqual(note1.tags, ["evolved_tag", "updated"])
        self.assertEqual(note1.keywords, ["evolved", "keyword"])

    def test_evolution_history_records_trigger(self):
        """Evolution history should record the trigger memory ID."""
        system = _create_system_with_mock()
        id1 = _add_memory_directly(system, "Base memory content")

        mock: MockLLMController = system.llm_controller.llm
        mock.analyze_response = json.dumps({
            "keywords": ["trigger"],
            "context": "Trigger",
            "tags": ["test"],
        })
        mock.link_response = json.dumps({
            "should_link": False,
            "connections": [],
            "link_reasons": [],
        })
        mock.evolution_responses = [
            json.dumps({
                "should_evolve": True,
                "new_context": "Updated",
                "new_tags": ["updated"],
                "new_keywords": ["updated"],
            }),
        ]

        id2 = system.add_note("Trigger memory content")

        note1 = system.read(id1)
        self.assertTrue(len(note1.evolution_history) > 0, "Should have evolution history")
        latest = note1.evolution_history[-1]
        self.assertEqual(latest["trigger"], id2)
        self.assertEqual(latest["action"], "evolution")
        self.assertIn("timestamp", latest)

    def test_non_evolved_neighbors_unchanged(self):
        """Neighbors where should_evolve=False should remain unchanged."""
        system = _create_system_with_mock()
        original_context = "Original unchanging context"
        original_tags = ["unchanging"]
        original_keywords = ["stable"]
        id1 = _add_memory_directly(system, "Stable content",
                                    keywords=original_keywords,
                                    context=original_context,
                                    tags=original_tags)

        mock: MockLLMController = system.llm_controller.llm
        mock.analyze_response = json.dumps({
            "keywords": ["other"],
            "context": "Other",
            "tags": ["other"],
        })
        mock.link_response = json.dumps({
            "should_link": False,
            "connections": [],
            "link_reasons": [],
        })
        mock.evolution_responses = [
            json.dumps({
                "should_evolve": False,
                "new_context": "This should not be applied",
                "new_tags": ["should_not_appear"],
                "new_keywords": ["should_not_appear"],
            }),
        ]

        id2 = system.add_note("Unrelated content")

        note1 = system.read(id1)
        self.assertEqual(note1.context, original_context)
        self.assertEqual(note1.tags, original_tags)
        self.assertEqual(note1.keywords, original_keywords)


class TestEvolutionPersistence(unittest.TestCase):
    """Test that evolution results are persisted to ChromaDB."""

    def test_evolved_metadata_in_chroma(self):
        system = _create_system_with_mock()
        id1 = _add_memory_directly(system, "Persist test content",
                                    keywords=["old"], context="Old context",
                                    tags=["old_tag"])

        mock: MockLLMController = system.llm_controller.llm
        mock.analyze_response = json.dumps({
            "keywords": ["new"],
            "context": "New",
            "tags": ["new"],
        })
        mock.link_response = json.dumps({
            "should_link": False,
            "connections": [],
            "link_reasons": [],
        })
        mock.evolution_responses = [
            json.dumps({
                "should_evolve": True,
                "new_context": "Persisted evolved context",
                "new_tags": ["persisted_tag"],
                "new_keywords": ["persisted_keyword"],
            }),
        ]

        id2 = system.add_note("Trigger content for persistence test")

        # Verify directly from ChromaDB via ID lookup (not semantic search)
        results = system.retriever.collection.get(ids=[id1], include=["metadatas"])
        self.assertTrue(len(results['ids']) > 0, f"id1 should exist in ChromaDB")
        meta = results['metadatas'][0]
        self.assertEqual(meta.get('context'), "Persisted evolved context")
        # tags/keywords may be stored as JSON strings
        tags = meta.get('tags', [])
        if isinstance(tags, str):
            tags = json.loads(tags)
        keywords = meta.get('keywords', [])
        if isinstance(keywords, str):
            keywords = json.loads(keywords)
        self.assertEqual(tags, ["persisted_tag"])
        self.assertEqual(keywords, ["persisted_keyword"])

    def test_consolidation_preserves_evolved_data(self):
        """After consolidation, evolved metadata should still be intact."""
        system = _create_system_with_mock()
        id1 = _add_memory_directly(system, "Content to evolve and consolidate",
                                    keywords=["pre_evo"], context="Pre evolution",
                                    tags=["pre"])

        mock: MockLLMController = system.llm_controller.llm
        mock.analyze_response = json.dumps({
            "keywords": ["trigger"],
            "context": "Trigger",
            "tags": ["trigger"],
        })
        mock.link_response = json.dumps({
            "should_link": True,
            "connections": ["0"],
            "link_reasons": ["related"],
        })
        mock.evolution_responses = [
            json.dumps({
                "should_evolve": True,
                "new_context": "Post evolution context",
                "new_tags": ["evolved"],
                "new_keywords": ["evolved"],
            }),
        ]

        id2 = system.add_note("Trigger for consolidation test")

        # Consolidate
        system.consolidate_memories()

        # Verify post-consolidation
        note1 = system.read(id1)
        self.assertEqual(note1.context, "Post evolution context")
        self.assertEqual(note1.tags, ["evolved"])

        # Also check links survived consolidation
        note2 = system.read(id2)
        # At minimum, either note should still have links
        total_links = len(note1.links) + len(note2.links)
        self.assertTrue(total_links > 0, "Links should survive consolidation")


class TestIndexMapping(unittest.TestCase):
    """Test that ID mapping works correctly (modification 6)."""

    def test_correct_memory_accessed_by_id(self):
        """With many memories, the correct one should be linked/evolved."""
        system = _create_system_with_mock()
        ids = []
        for i in range(10):
            mid = _add_memory_directly(
                system,
                f"Memory content number {i} about topic {chr(65+i)}",
                keywords=[f"topic_{chr(65+i)}"],
                tags=[f"tag_{i}"],
            )
            ids.append(mid)

        mock: MockLLMController = system.llm_controller.llm
        mock.analyze_response = json.dumps({
            "keywords": ["topic_A"],
            "context": "Related to topic A",
            "tags": ["related"],
        })
        # Link to index 0 in search results
        mock.link_response = json.dumps({
            "should_link": True,
            "connections": ["0"],
            "link_reasons": ["directly related"],
        })
        # All neighbors: don't evolve (simplify test)
        mock.evolution_responses = [
            json.dumps({
                "should_evolve": False,
                "new_context": "",
                "new_tags": [],
                "new_keywords": [],
            })
        ] * 5  # up to evolution_k

        new_id = system.add_note("New content related to topic A")
        new_note = system.read(new_id)

        # Verify that links point to actual valid memory IDs
        for link_id in new_note.links:
            self.assertIn(
                link_id, system.memories,
                f"Link {link_id} should be a valid memory ID"
            )
            # Should NOT be the new note itself
            self.assertNotEqual(link_id, new_id)


class TestEvolutionKParameter(unittest.TestCase):
    """Test that evolution_k parameter controls the number of neighbors used."""

    def test_evolution_k_limits_neighbors(self):
        system = _create_system_with_mock(evolution_k=2)
        for i in range(5):
            _add_memory_directly(system, f"Content {i}")

        mock: MockLLMController = system.llm_controller.llm
        mock.analyze_response = json.dumps({
            "keywords": ["test"],
            "context": "Test",
            "tags": ["test"],
        })
        mock.link_response = json.dumps({
            "should_link": False,
            "connections": [],
            "link_reasons": [],
        })
        mock.evolution_responses = [
            json.dumps({
                "should_evolve": False,
                "new_context": "",
                "new_tags": [],
                "new_keywords": [],
            })
        ] * 2

        mock.reset_call_tracking()
        system.add_note("Test query content")

        evolution_calls = [
            c for c in mock.call_history
            if "memory evolution agent" in c["prompt"]
        ]
        # With evolution_k=2, we should have at most 2 evolution calls
        self.assertLessEqual(
            len(evolution_calls), 2,
            f"With evolution_k=2, should have at most 2 evolution calls, got {len(evolution_calls)}"
        )

    def test_default_evolution_k_is_5(self):
        system = AgenticMemorySystem(
            model_name="all-MiniLM-L6-v2",
            llm_backend="openai",
            llm_model="gpt-4o-mini",
        )
        self.assertEqual(system.evolution_k, 5)


class TestPersistMemoryToChroma(unittest.TestCase):
    """Test the _persist_memory_to_chroma helper."""

    def test_persist_updates_chroma(self):
        system = _create_system_with_mock()
        id1 = _add_memory_directly(system, "Original content",
                                    context="Original", tags=["old"])

        note = system.memories[id1]
        note.context = "Updated context"
        note.tags = ["new_tag"]
        system._persist_memory_to_chroma(note)

        # Verify from ChromaDB
        results = system.retriever.search("Original content", k=1)
        self.assertTrue(len(results['ids'][0]) > 0)
        meta = results['metadatas'][0][0]
        self.assertEqual(meta.get('context'), "Updated context")
        self.assertEqual(meta.get('tags'), ["new_tag"])

    def test_persist_handles_missing_document(self):
        """Should not raise even if document doesn't exist in ChromaDB yet."""
        system = _create_system_with_mock()
        note = MemoryNote(content="Brand new content", keywords=["test"])
        # This should not raise (delete will fail silently, then add)
        system._persist_memory_to_chroma(note)
        results = system.retriever.search("Brand new content", k=1)
        self.assertTrue(len(results['ids'][0]) > 0)


class TestFormatNeighborsForPrompt(unittest.TestCase):
    """Test the _format_neighbors_for_prompt helper."""

    def test_formats_existing_memories(self):
        system = _create_system_with_mock()
        id1 = _add_memory_directly(system, "Content A", keywords=["a"], tags=["tag_a"])
        id2 = _add_memory_directly(system, "Content B", keywords=["b"], tags=["tag_b"])

        text = system._format_neighbors_for_prompt([id1, id2])
        self.assertIn("Content A", text)
        self.assertIn("Content B", text)
        self.assertIn(id1, text)
        self.assertIn(id2, text)

    def test_returns_none_for_empty_list(self):
        system = _create_system_with_mock()
        text = system._format_neighbors_for_prompt([])
        self.assertEqual(text, "(none)")

    def test_skips_missing_ids(self):
        system = _create_system_with_mock()
        id1 = _add_memory_directly(system, "Existing content")
        text = system._format_neighbors_for_prompt([id1, "nonexistent-id"])
        self.assertIn("Existing content", text)
        self.assertNotIn("nonexistent-id", text)


class TestPromptSeparation(unittest.TestCase):
    """Test that Ps2 and Ps3 are separate LLM calls."""

    def test_separate_link_and_evolution_calls(self):
        """process_memory should make separate calls for linking and evolution."""
        system = _create_system_with_mock()
        id1 = _add_memory_directly(system, "Existing content 1")
        id2 = _add_memory_directly(system, "Existing content 2")

        mock: MockLLMController = system.llm_controller.llm
        mock.link_response = json.dumps({
            "should_link": True,
            "connections": ["0"],
            "link_reasons": ["related"],
        })
        mock.evolution_responses = [
            json.dumps({
                "should_evolve": False,
                "new_context": "",
                "new_tags": [],
                "new_keywords": [],
            }),
            json.dumps({
                "should_evolve": False,
                "new_context": "",
                "new_tags": [],
                "new_keywords": [],
            }),
        ]

        note = MemoryNote(content="New content", keywords=["test"], context="Test", tags=["test"])
        mock.reset_call_tracking()

        system.process_memory(note)

        link_calls = [c for c in mock.call_history if "memory linking agent" in c["prompt"]]
        evo_calls = [c for c in mock.call_history if "memory evolution agent" in c["prompt"]]

        self.assertEqual(len(link_calls), 1, "Should have exactly 1 link generation call (Ps2)")
        self.assertEqual(len(evo_calls), 2, "Should have 2 evolution calls (Ps3, one per neighbor)")

    def test_no_old_evolution_prompt_used(self):
        """The old single _evolution_system_prompt should not exist."""
        system = _create_system_with_mock()
        self.assertFalse(
            hasattr(system, '_evolution_system_prompt'),
            "Old _evolution_system_prompt should be removed"
        )
        self.assertTrue(hasattr(system, '_link_generation_prompt'))
        self.assertTrue(hasattr(system, '_evolution_prompt'))


class TestProcessMemoryIntegration(unittest.TestCase):
    """Integration test for the full process_memory flow."""

    def test_full_flow_link_and_evolve(self):
        """Test that a full add_note triggers both linking and evolution."""
        system = _create_system_with_mock()
        id1 = _add_memory_directly(
            system, "Deep learning fundamentals",
            keywords=["deep_learning"], context="AI basics", tags=["AI"]
        )
        id2 = _add_memory_directly(
            system, "Backpropagation algorithm",
            keywords=["backprop"], context="Training methods", tags=["AI", "math"]
        )

        mock: MockLLMController = system.llm_controller.llm
        mock.analyze_response = json.dumps({
            "keywords": ["gradient", "descent"],
            "context": "Optimization in deep learning",
            "tags": ["AI", "optimization"],
        })
        mock.link_response = json.dumps({
            "should_link": True,
            "connections": ["0", "1"],
            "link_reasons": ["related to DL", "related to training"],
        })
        # Both neighbors evolve with the same response (order-independent)
        evo_response = json.dumps({
            "should_evolve": True,
            "new_context": "Evolved context with gradient descent",
            "new_tags": ["AI", "optimization"],
            "new_keywords": ["evolved", "gradient"],
        })
        mock.evolution_responses = [evo_response, evo_response]

        id3 = system.add_note("Gradient descent optimization techniques")

        # Check new note has links
        note3 = system.read(id3)
        self.assertTrue(len(note3.links) > 0, "New note should have links")

        # Check both neighbors were evolved (order-independent assertion)
        note1 = system.read(id1)
        note2 = system.read(id2)
        evolved_contexts = {note1.context, note2.context}
        self.assertIn(
            "Evolved context with gradient descent", evolved_contexts,
            "At least one neighbor should have evolved context"
        )

        # Check both have "optimization" tag
        for note in [note1, note2]:
            if note.context == "Evolved context with gradient descent":
                self.assertIn("optimization", note.tags)

        # Check bidirectional links
        for linked_id in note3.links:
            linked = system.read(linked_id)
            if linked:
                self.assertIn(
                    id3, linked.links,
                    f"Reverse link missing: {linked_id} -> {id3}"
                )

        # Check evolution history exists on at least one neighbor
        all_histories = note1.evolution_history + note2.evolution_history
        self.assertTrue(len(all_histories) > 0, "Should have evolution history")
        triggers = [h["trigger"] for h in all_histories]
        self.assertIn(id3, triggers)

    def test_first_memory_skips_evolution(self):
        """First memory added should skip evolution."""
        system = _create_system_with_mock()
        mock: MockLLMController = system.llm_controller.llm
        mock.analyze_response = json.dumps({
            "keywords": ["first"],
            "context": "First memory",
            "tags": ["first"],
        })

        mock.reset_call_tracking()
        id1 = system.add_note("Very first memory in the system")

        # No link or evolution calls should have been made
        link_calls = [c for c in mock.call_history if "memory linking agent" in c["prompt"]]
        evo_calls = [c for c in mock.call_history if "memory evolution agent" in c["prompt"]]
        self.assertEqual(len(link_calls), 0)
        self.assertEqual(len(evo_calls), 0)


if __name__ == "__main__":
    unittest.main()
