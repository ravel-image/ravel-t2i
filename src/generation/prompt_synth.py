"""
src/generation/prompt_synth.py
───────────────────────────────
Contrastive Chain-of-Thought prompt synthesis (Section 3.2, Table 11).

Implements the paper's exact 3-step pipeline:

    Step 1 — Resolve all subjects from query + KG
              "Yama and his sister" → Yama, Yami

    Step 2 — Per-entity contrastive sub-prompt covering all 8 attribute
              categories from Figure 16:
                - Unique Facts
                - Functional Properties
                - Physical Properties
                - Taxonomic Info (biology) / Personality Traits (mythology)
                - Material & Composition
                - Usage Context
                - Origin & History
                - Distinctive Features (NOT-X constraints)

    Step 3 — CoT merge: fuse all sub-prompts into one coherent scene prompt
              that expresses ALL entities with equal visual fidelity

Paper reference: Section 3.2, Table 11, Figure 16
"""

import os
import logging
from dataclasses import dataclass, field

from openai import OpenAI
from src.kg.retriever import ContextPacket

logger = logging.getLogger(__name__)


@dataclass
class EnrichedPrompt:
    original:         str
    enriched:         str
    entity_prompts:   dict[str, str] = field(default_factory=dict)  # per-entity sub-prompts
    contrastive_cues: list[str]      = field(default_factory=list)


class PromptSynthesizer:
    """
    Three-step contrastive CoT synthesizer matching Table 11 of the paper.

    Step 2 generates one contrastive sub-prompt per entity covering all
    8 attribute categories from Figure 16. Step 3 merges them into one
    scene with correct relational composition.
    """

    # ── Step 2: per-entity contrastive sub-prompt ────────────────────────────

    _ENTITY_SYSTEM = """\
You write a precise contrastive visual sub-prompt for a single rare entity
for use in a text-to-image generation pipeline.

Cover ALL of the following attribute categories (skip only if genuinely N/A):

1. PHYSICAL PROPERTIES — exact body structure, number of arms/heads/limbs,
   overall form, size, proportions
2. APPEARANCE / MORPHOLOGY — colors (be specific: "storm-cloud dark blue",
   not "dark"), texture (smooth/scaly/feathered), surface quality
3. UNIQUE FACTS — what makes this entity visually unlike anything common
4. FUNCTIONAL PROPERTIES — what it does, its role, powers, ecological function
5. PERSONALITY TRAITS / BEHAVIOR — emotional expression, posture, demeanor
   (important for mythological figures and animals)
6. MATERIAL & COMPOSITION — what it is made of or composed of (for artifacts)
   or biological makeup (for organisms)
7. USAGE CONTEXT / SYMBOLIC ITEMS — what it carries, wears, is associated with
8. ORIGIN & HISTORY — cultural/geographic grounding, era, tradition
9. DISTINCTIVE FEATURES with NOT-X constraints — at least 2 explicit
   "NOT [generic alternative]" phrases that prevent the model defaulting
   to a common prior. Example: "NOT a western grim reaper", "NOT a generic deer"

Write a dense, vivid paragraph (60-100 words) covering as many of these
as possible. Do NOT use numbered lists — write as flowing descriptive prose.
Return ONLY the sub-prompt text."""

    _ENTITY_USER = """\
Entity: {name}
Domain: {domain}
Type: {entity_type}

Physical / Visual attributes:
  Morphology: {morphology}
  Distinctive features: {distinctive_features}
  Colors: {colors}
  Texture: {texture}
  Size: {size}
  Structure: {structure}

Functional / Contextual:
  Primary function: {primary_function}
  Origin: {origin}
  Cultural significance: {significance}
  Historical period: {period}

Contrastive constraints (must appear as NOT-X in your sub-prompt):
{contrastive}

Write the contrastive sub-prompt for {name} now."""

    # ── Step 3: CoT merge ─────────────────────────────────────────────────────

    _MERGE_SYSTEM = """\
You are composing a final text-to-image prompt by merging per-entity
contrastive sub-prompts into one coherent scene.

Rules:
1. Every entity sub-prompt below must be fully represented — do not drop
   any entity's visual details or NOT-X constraints
2. Describe how the entities relate to each other visually in the scene,
   based on their relationship type
3. Maintain the original query's intent — if the query asks for
   "Yama and his sister", both Yama AND Yami must be prominent
4. Compose as ONE flowing paragraph of vivid, specific imagery
5. Keep all NOT-X constraints from all sub-prompts
6. Do NOT add generic quality tags like "8k masterpiece" or "photorealistic"
   unless they are domain-appropriate

Return ONLY the final merged prompt."""

    _MERGE_USER = """\
ORIGINAL QUERY: "{query}"

RELATIONSHIP CONTEXT:
{relationship_block}

PER-ENTITY SUB-PROMPTS:
{sub_prompts_block}

Merge these into one coherent scene prompt that captures all entities
as described above with their relationship visually expressed."""

    # ── SRD refinement ───────────────────────────────────────────────────────

    _REFINEMENT_SYSTEM = """\
You refine a text-to-image prompt to fix missing visual attributes.

Given the current prompt and a list of attributes NOT visible in the
last generated image, revise the prompt to strongly emphasise those
missing attributes WITHOUT removing correctly rendered ones.

Apply corrections proportionally to the decay weight:
    1.0 = strong emphasis, 0.5 = moderate, lower = subtle

Return ONLY the revised prompt — no preamble, no explanation."""

    _REFINEMENT_USER = """\
CURRENT PROMPT:
{current_prompt}

MISSING ATTRIBUTES (emphasise these strongly):
{missing_attrs}

DECAY WEIGHT: {decay:.2f}

Write the refined prompt now."""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ── Public: synthesize ────────────────────────────────────────────────────

    def synthesize(self, ctx: ContextPacket) -> EnrichedPrompt:
        """
        Three-step contrastive CoT synthesis per Table 11 of the paper.

        Step 1: entities already resolved by retriever (primary_entities)
        Step 2: generate per-entity contrastive sub-prompt
        Step 3: CoT merge into final scene prompt
        """
        if ctx.is_empty():
            logger.warning("Empty ContextPacket — returning original prompt unchanged.")
            return EnrichedPrompt(original=ctx.query, enriched=ctx.query)

        # Step 2 — per-entity contrastive sub-prompts
        entity_prompts: dict[str, str] = {}
        for entity in ctx.primary_entities:
            name = entity.get("name", "")
            if not name:
                continue
            sub = self._generate_entity_subprompt(entity)
            entity_prompts[name] = sub
            logger.info(f"  Sub-prompt for '{name}': {sub[:80]}...")

        if not entity_prompts:
            return EnrichedPrompt(original=ctx.query, enriched=ctx.query)

        # Step 3 — CoT merge
        relationship_block = self._build_relationship_block(ctx)
        sub_prompts_block  = "\n\n".join(
            f"[{name}]\n{sub}" for name, sub in entity_prompts.items()
        )

        merge_user = self._MERGE_USER.format(
            query=ctx.query,
            relationship_block=relationship_block,
            sub_prompts_block=sub_prompts_block,
        )

        enriched = self._call_llm(
            system=self._MERGE_SYSTEM,
            user=merge_user,
            max_tokens=600,
        )

        logger.info(f"  Enriched prompt synthesised ({len(enriched)} chars)")

        return EnrichedPrompt(
            original=ctx.query,
            enriched=enriched,
            entity_prompts=entity_prompts,
            contrastive_cues=ctx.contrastive_constraints,
        )

    # ── Public: refine (SRD) ─────────────────────────────────────────────────

    def refine(
        self,
        current_prompt:     str,
        missing_attributes: list[str],
        decay:              float,
        round_idx:          int,
    ) -> str:
        if not missing_attributes:
            return current_prompt

        missing_str = "\n".join(f"- {a}" for a in missing_attributes)
        user_msg = self._REFINEMENT_USER.format(
            current_prompt=current_prompt,
            missing_attrs=missing_str,
            decay=decay,
        )

        refined = self._call_llm(
            system=self._REFINEMENT_SYSTEM,
            user=user_msg,
            max_tokens=600,
        )

        logger.info(
            f"  SRD Round {round_idx}: refined prompt "
            f"({len(missing_attributes)} missing attrs, decay={decay:.2f})"
        )
        return refined

    # ── Step 2 helper ─────────────────────────────────────────────────────────

    def _generate_entity_subprompt(self, entity: dict) -> str:
        """Generate one contrastive sub-prompt for a single entity."""
        visual      = {}
        functional  = {}
        contextual  = {}

        # Support both flat (loaded from Neo4j) and nested (raw JSON) formats
        if "visual_attributes" in entity:
            visual     = entity.get("visual_attributes", {})
            functional = entity.get("functional_attributes", {})
            contextual = entity.get("contextual_attributes", {})
        else:
            visual     = entity  # flat Neo4j node
            contextual = entity
            functional = entity

        name = entity.get("name", "")
        contrastive_list = entity.get("contrastive_constraints", []) or []
        contrastive_str  = "\n".join(f"  - {c}" for c in contrastive_list) \
                           if contrastive_list else "  (derive from domain context)"

        user_msg = self._ENTITY_USER.format(
            name=name,
            domain=entity.get("domain", ""),
            entity_type=entity.get("entity_type", ""),
            morphology=visual.get("morphology", entity.get("morphology", "")),
            distinctive_features="; ".join(
                visual.get("distinctive_features", entity.get("distinctive_features", [])) or []
            ),
            colors=", ".join(
                visual.get("color_palette", entity.get("color_palette", [])) or []
            ),
            texture=visual.get("texture", entity.get("texture", "")),
            size=visual.get("size_and_scale", entity.get("size_and_scale", "")),
            structure=visual.get("structural_arrangement", entity.get("structural_arrangement", "")),
            primary_function=functional.get("primary_function", entity.get("primary_function", "")),
            origin=contextual.get("origin", entity.get("origin", "")),
            significance=contextual.get("cultural_significance", entity.get("cultural_significance", "")),
            period=contextual.get("historical_period", entity.get("historical_period", "")),
            contrastive=contrastive_str,
        )

        return self._call_llm(
            system=self._ENTITY_SYSTEM,
            user=user_msg,
            max_tokens=250,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_relationship_block(self, ctx: ContextPacket) -> str:
        """Format the most relevant relationships for the merge step."""
        if not ctx.relationships:
            return "No explicit relationships retrieved."

        # Filter to relationships between primary entities only
        primary_names = {e.get("name") for e in ctx.primary_entities}
        relevant = [
            r for r in ctx.relationships
            if r.get("from") in primary_names and r.get("to") in primary_names
        ]

        # Fall back to all relationships if none between primaries
        if not relevant:
            relevant = ctx.relationships[:6]

        lines = [f"  ({r['from']}) --[{r['type']}]--> ({r['to']})" for r in relevant[:8]]
        return "\n".join(lines)

    def _call_llm(self, system: str, user: str, max_tokens: int = 512) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()
