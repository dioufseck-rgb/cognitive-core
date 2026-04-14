"""
Cognitive Core — Causal DAG Loader and Serializer (TASK 5)

Loads causal DAG structure files from disk and serializes them for LLM prompts.
The DAG is passed as structured JSON in a causal_context block — never as prose.
The LLM is read-only with respect to the DAG; it cannot modify the structure.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("cognitive_core.analytics.causal_dag")


# ── Exceptions ────────────────────────────────────────────────────────────────

class DAGLoadError(RuntimeError):
    """Raised when a DAG structure file cannot be loaded or is malformed."""


# ── Loader ────────────────────────────────────────────────────────────────────

class CausalDAGLoader:
    """
    Loads and validates a causal DAG from a JSON structure file.

    The DAG is read-only at runtime. The LLM receives a serialized copy
    of the DAG but cannot alter the source structure.
    """

    REQUIRED_KEYS = {"nodes", "edges"}

    def __init__(self, structure_file: str, base_dir: str = "."):
        path = Path(structure_file)
        if not path.is_absolute():
            path = Path(base_dir) / path
        self._path = path
        self._dag: dict[str, Any] | None = None

    def load(self) -> dict[str, Any]:
        """Load and validate the DAG. Returns the DAG dict. Raises DAGLoadError on failure."""
        if self._dag is not None:
            return self._dag

        if not self._path.exists():
            raise DAGLoadError(f"DAG structure file not found: {self._path}")

        try:
            with open(self._path) as f:
                dag = json.load(f)
        except json.JSONDecodeError as e:
            raise DAGLoadError(f"DAG file is not valid JSON: {self._path}: {e}") from e

        missing = self.REQUIRED_KEYS - set(dag.keys())
        if missing:
            raise DAGLoadError(
                f"DAG file missing required keys {missing}: {self._path}"
            )

        self._dag = dag
        return dag

    @property
    def dag_id(self) -> str:
        dag = self.load()
        return dag.get("dag_id", str(self._path.stem))

    @property
    def version(self) -> str:
        dag = self.load()
        return dag.get("version", "1.0")


# ── Serializer ────────────────────────────────────────────────────────────────

def serialize_dag_for_prompt(dag: dict[str, Any]) -> str:
    """
    Serialize a causal DAG as structured JSON for inclusion in an LLM prompt.

    The serialized form is placed in a causal_context block that instructs
    the LLM to reason within the DAG structure without modifying it.
    """
    # Include the full DAG structure for LLM reasoning
    return json.dumps(dag, indent=2)


def build_causal_context_block(
    dag: dict[str, Any],
    artifact_name: str,
    dag_origin: str = "pre_existing",
) -> str:
    """
    Build the causal_context prompt block injected into the investigate prompt.

    The block tells the LLM:
    - What the DAG is and where it came from
    - The graph structure (nodes, edges, paths)
    - What it should output (activated paths, unobserved nodes, etc.)
    """
    dag_json = serialize_dag_for_prompt(dag)
    return f"""
=== CAUSAL CONTEXT (READ-ONLY — DO NOT MODIFY) ===
Artifact: {artifact_name}
DAG ID: {dag.get('dag_id', 'unknown')}
Origin: {dag_origin}
Description: {dag.get('description', '')}

CAUSAL DAG STRUCTURE:
{dag_json}

CAUSAL REASONING INSTRUCTIONS:
Using the DAG structure above, you MUST extend your investigation output with:
1. activated_paths: Which DAG paths are active given the evidence? List path_ids from the DAG.
2. alternative_paths_considered: Which paths were considered but not activated? Explain why.
3. unobserved_nodes: Which DAG nodes have no evidence in the case data? List node IDs.
4. evidential_gaps: What evidence would be needed to confirm or rule out each activated path?
5. dag_divergence_flag: If the case evidence suggests a causal structure NOT captured in the DAG, set this to true and explain in integration_reasoning.
6. integration_reasoning: Your reasoning about how the evidence maps to the DAG structure.
7. causal_templates_invoked: List the path_ids from the DAG that were evaluated.
8. dag_version: The dag_id from the DAG above.

Extend your JSON response with these additional fields alongside the standard fields.
=== END CAUSAL CONTEXT ===
"""


# ── Convenience loader ────────────────────────────────────────────────────────

def load_dag_for_artifact(
    artifact: dict[str, Any],
    base_dir: str = ".",
) -> dict[str, Any] | None:
    """
    Load a causal DAG given an artifact config dict from the registry.

    Returns the DAG dict or None if loading fails (caller should fallback to v1).
    """
    dag_config = artifact.get("dag_config", {})
    structure_file = dag_config.get("structure_file", "")
    if not structure_file:
        logger.warning("Artifact '%s' has no dag_config.structure_file",
                       artifact.get("artifact_name", "?"))
        return None

    try:
        loader = CausalDAGLoader(structure_file, base_dir=base_dir)
        return loader.load()
    except DAGLoadError as e:
        logger.warning("Failed to load DAG for artifact '%s': %s",
                       artifact.get("artifact_name", "?"), e)
        return None
