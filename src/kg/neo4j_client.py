"""
src/kg/neo4j_client.py
───────────────────────
Thin Neo4j driver wrapper.

Handles:
    - Connection and authentication
    - Running Cypher queries
    - Ensuring uniqueness constraints on startup
"""

import os
import logging
from typing import Any

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Neo4j driver wrapper.

    Reads connection details from environment variables:
        NEO4J_URI       (default: bolt://localhost:7687)
        NEO4J_USER      (default: neo4j)
        NEO4J_PASSWORD  (required)

    Usage:
        client = Neo4jClient()
        client.run("MATCH (e:Entity) RETURN e.name LIMIT 5")
        client.close()

    Or as a context manager:
        with Neo4jClient() as client:
            client.run(cypher, params)
    """

    def __init__(self):
        uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
        user     = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")

        if not password:
            raise ValueError(
                "NEO4J_PASSWORD environment variable is not set."
            )

        try:
            self._driver: Driver = GraphDatabase.driver(
                uri, auth=(user, password)
            )
            self._driver.verify_connectivity()
            logger.info(f"Neo4j connected: {uri}")

        except AuthError:
            raise ConnectionError(
                f"Neo4j authentication failed for user '{user}'. "
                "Check NEO4J_USER and NEO4J_PASSWORD."
            )
        except ServiceUnavailable:
            raise ConnectionError(
                f"Cannot reach Neo4j at '{uri}'. "
                "Make sure the database is running."
            )

    # ── Core query interface ──────────────────────────────────────────────────

    def run(self, cypher: str, params: dict | None = None) -> list[dict[str, Any]]:
        """
        Execute a Cypher query and return results as a list of dicts.

        Args:
            cypher : Cypher query string
            params : optional parameters (use $param_name in Cypher)

        Returns:
            List of record dicts. Empty list if no rows returned.
        """
        params = params or {}
        with self._driver.session() as session:
            result = session.run(cypher, **params)
            return [record.data() for record in result]

    # ── Schema setup ──────────────────────────────────────────────────────────

    def ensure_constraints(self) -> None:
        """
        Create uniqueness constraint on Entity.name.
        Called once at the start of the loading pipeline.
        Safe to call multiple times — uses IF NOT EXISTS.
        """
        cypher = """
        CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
        FOR (e:Entity) REQUIRE e.name IS UNIQUE
        """
        try:
            self.run(cypher)
            logger.info("Neo4j: uniqueness constraint on Entity.name ensured.")
        except Exception as e:
            # Some Neo4j community editions don't support IF NOT EXISTS
            logger.warning(f"Neo4j: could not create constraint (may already exist): {e}")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._driver.close()
        logger.info("Neo4j: driver closed.")

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
