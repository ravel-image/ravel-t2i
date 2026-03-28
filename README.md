# RAVEL: Rare Concept Generation and Editing via Graph-driven Relational Guidance

A training-free framework that grounds text-to-image synthesis in a structured
Knowledge Graph (KG) to enable high-fidelity generation of rare, culturally
nuanced, and long-tail concepts.

---


## Quick Start — Generate Images with RAVEL

**Prerequisites:** 
1) `.env` configured, Neo4j instance running (you can get a free auroradb instance), KG built (see Part 1 and Option B below).
2) Please ensure your venv is running on at least Python=3.12

### Step 1 — Build the KG (one time only)

```bash
python scripts/build_kg.py --all
```

This scrapes Wikipedia for all entities across all 7 domains, extracts
structured attributes via GPT-4o, and loads everything into Neo4j.
Takes ~15–30 minutes. Only needs to run once.

### Step 2 — Generate an image

```bash
# DALL-E 3 (no GPU needed) + SRD self-correction
python scripts/run_generation.py \
    --prompt "Hindu god Yama seated on a water buffalo" \
    --backbone dalle3 \
    --srd \
    --output output/

# Flux (GPU) + SRD
python scripts/run_generation.py \
    --prompt "Saola standing in dense Vietnamese forest" \
    --backbone flux \
    --srd \
    --output output/

# SDXL (GPU) + SRD
python scripts/run_generation.py \
    --prompt "Kapala skull bowl Tibetan ritual object" \
    --backbone sdxl \
    --srd \
    --output output/

# Janus-Pro autoregressive (GPU) + SRD
python scripts/run_generation.py \
    --prompt "Zulu Isicholo traditional headdress" \
    --backbone janus_pro \
    --srd \
    --output output/

# Without SRD — RAVEL enrichment only, no iterative correction
python scripts/run_generation.py \
    --prompt "Ganda Bherunda two-headed mythical bird" \
    --backbone dalle3 \
    --output output/
```

### Step 3 — View results

Each run creates a dedicated folder under `output/`:

```
output/yama_dalle3/
├── 00_base.png              ← vanilla model, raw prompt, no RAVEL
├── 01_ravel.png             ← KG-enriched prompt, no SRD
├── 02_srd_r1_gsi0.22.png   ← after SRD round 1
├── 03_srd_r2_gsi1.00.png   ← after SRD round 2 (converged)
├── final.png                ← best image
└── run_info.json            ← prompts, GSI scores, attributes used
```

Compare `00_base.png` → `01_ravel.png` → `final.png` to see:
- `00_base` : what the vanilla model produces with no guidance
- `01_ravel`: effect of KG-enriched contrastive prompting
- `final`   : effect of iterative SRD self-correction

---

---

## Project Structure

```
ravel/
├── src/
│   ├── data/
│   │   ├── prompts.py                  # Universal LLM extraction prompt
│   │   └── sample_entities/            # Pre-made entity lists per domain
│   │       ├── indian_mythology.json
│   │       ├── greek_mythology.json
│   │       ├── chinese_mythology.json
│   │       ├── literary.json
│   │       ├── biology.json
│   │       ├── natural_phenomena.json
│   │       └── cultural_artifact.json
│   └── kg/
│       ├── entity_generator.py         # LLM auto-generates rare entity names
│       ├── scraper.py                  # Wikipedia + Gutenberg scraping
│       ├── extractor.py                # LLM structured attribute extraction
│       ├── loader.py                   # Neo4j node + edge loading
│       ├── neo4j_client.py             # Neo4j driver wrapper
│       └── retriever.py                # k-hop subgraph retrieval
│   ├── generation/
│   │   ├── prompt_synth.py             # Contrastive CoT prompt synthesis
│   │   └── backbone.py                 # SDXL / Flux / DALL-E 3 / Janus-Pro / GLM-Image
│   └── srd/
│       ├── verifier.py                 # VLM binary attribute verification (GSI)
│       └── refiner.py                  # SRD iterative self-correction loop
├── pipeline.py                         # Top-level orchestrator
├── scripts/
│   ├── build_kg.py                     # CLI: KG construction
│   └── run_generation.py               # CLI: image generation
└── data/
    ├── output/                         # Extracted entity JSONs per domain
    └── extracted_entities/             # Auto-generated entity lists
```

---

## Setup

### 1. Install dependencies

```bash
pip install openai neo4j wikipedia-api wikipedia beautifulsoup4 lxml \
            requests python-dotenv diffusers transformers accelerate \
            torch torchvision
```

### 2. Environment variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_aura_password
HF_TOKEN=hf_...        # required for Flux and Janus-Pro
```

**Neo4j:** Create a free AuraDB instance at console.neo4j.io.
The free tier pauses after 3 days of inactivity — resume it from the console
before running if needed.

**HuggingFace token:** Required for gated models (Flux, Janus-Pro).
Create one at huggingface.co → Settings → Access Tokens.

### 3. Verify connections

```bash
# Test Neo4j
python -c "
from dotenv import load_dotenv; load_dotenv()
from src.kg.neo4j_client import Neo4jClient
with Neo4jClient() as c: print('Neo4j: OK')
"

# Test OpenAI
python -c "
import os; from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI
r = OpenAI(api_key=os.getenv('OPENAI_API_KEY')).chat.completions.create(
    model='gpt-4o', messages=[{'role':'user','content':'ping'}], max_tokens=5)
print('OpenAI: OK')
"
```

---

## Part 1 — Knowledge Graph Construction

RAVEL uses a Neo4j knowledge graph storing structured entity attributes
(visual, functional, relational, contextual) extracted from Wikipedia.

### Option A — Use pre-made entity lists (recommended for replication)

Processes the entity lists in `src/data/sample_entities/`:

```bash
# Single domain
python scripts/build_kg.py --domain indian_mythology

# Multiple domains
python scripts/build_kg.py --domain indian_mythology greek_mythology biology

# All 7 domains at once
python scripts/build_kg.py --all
```

### Option B — Auto-generate entity lists via LLM

The LLM generates n rare entities for a domain, then extracts and loads them:

```bash
# Generate 20 rare biology entities automatically
python scripts/build_kg.py --domain biology --auto-generate 20

# With custom authoritative source URLs
python scripts/build_kg.py --domain cultural_artifact --auto-generate 15 \
    --sources https://www.metmuseum.org https://www.britishmuseum.org

# Any domain name works — not limited to pre-defined ones
python scripts/build_kg.py --domain japanese_mythology --auto-generate 10
```

### Pipeline control flags

```bash
# Extract only (scrape + LLM) — inspect JSONs before loading
python scripts/build_kg.py --domain biology --extract-only

# Load only — JSONs already extracted, just load into Neo4j
python scripts/build_kg.py --domain biology --load-only

# Custom entity list
python scripts/build_kg.py --domain biology --entities path/to/my_list.json
```

### Verify the graph

Run these in the AuraDB Query console (console.neo4j.io):

```cypher
# Count entities per domain
MATCH (e:Entity) RETURN e.domain, count(e) AS count ORDER BY count DESC

# View nodes and edges
MATCH (a:Entity)-[r]->(b:Entity) RETURN a.name, type(r), b.name LIMIT 25

# Check attributes for a specific entity
MATCH (e:Entity {name: "Yama"})
RETURN e.morphology, e.distinctive_features, e.contrastive_constraints
```

---

## Part 2 — Image Generation

### Supported backbones

| Backbone | Type | Notes |
|---|---|---|
| `sdxl` | Diffusion U-Net | Local GPU, public |
| `flux` | Diffusion MM-DiT | Local GPU, needs HF_TOKEN |
| `dalle3` | Diffusion prior | OpenAI API, no GPU needed |
| `janus_pro` | Autoregressive | Local GPU, needs HF_TOKEN |
| `glm_image` | AR + DiT hybrid | Local GPU, 40GB+ VRAM |

### Basic generation

```bash
# DALL-E 3 — no GPU needed, good for testing
python scripts/run_generation.py \
    --prompt "Hindu god Yama seated on a water buffalo" \
    --backbone dalle3

# Flux
python scripts/run_generation.py \
    --prompt "Red Ginger plant blooming in tropical forest" \
    --backbone flux \
    --guidance-scale 3.5 --steps 50

# SDXL
python scripts/run_generation.py \
    --prompt "Saola standing in dense Vietnamese forest" \
    --backbone sdxl \
    --guidance-scale 7.5 --steps 50 --no-refiner

# Janus-Pro (autoregressive)
python scripts/run_generation.py \
    --prompt "Kapala skull bowl Tibetan ritual object" \
    --backbone janus_pro \
    --temperature 1.0 --cfg-weight 5.0

# GLM-Image (hybrid AR + diffusion)
python scripts/run_generation.py \
    --prompt "Zulu Isicholo traditional headdress" \
    --backbone glm_image \
    --guidance-scale 5.0 --steps 30
```

### Generation with SRD (iterative self-correction)

Add `--srd` to enable the Self-Correcting RAG-Guided Diffusion module:

```bash
python scripts/run_generation.py \
    --prompt "Hindu god Yama seated on a water buffalo" \
    --backbone dalle3 \
    --srd \
    --seed 42 \
    --output output/
```

SRD hyperparameters (paper defaults):

```bash
python scripts/run_generation.py \
    --prompt "Hindu god Yama seated on a water buffalo" \
    --backbone dalle3 \
    --srd \
    --tau 0.85 \       # GSI convergence threshold
    --max-k 3 \        # max SRD iterations
    --seed 42 \
    --output output/
```

### Batch generation from file

```bash
# Create a prompts file
cat > prompts.txt << 'END'
Hindu god Yama seated on a water buffalo
Saola standing in dense Vietnamese forest
Kapala skull bowl Tibetan ritual object
Zulu Isicholo traditional headdress
Red Ginger plant blooming in tropical forest
END

python scripts/run_generation.py \
    --prompts-file prompts.txt \
    --backbone dalle3 \
    --srd \
    --output output/batch/
```

### Output structure

Each run creates a folder per entity:

```
output/
└── yama_dalle3/
    ├── 00_base.png              # vanilla backbone, raw prompt, no KG
    ├── 01_ravel.png             # KG-enriched prompt, no SRD
    ├── 02_srd_r1_gsi0.22.png   # after SRD round 1
    ├── 03_srd_r2_gsi1.00.png   # after SRD round 2 (converged)
    ├── final.png                # best image
    └── run_info.json            # full metadata
```

`run_info.json` contains:
- Original and enriched prompts
- Retrieved KG attributes
- Contrastive constraints used
- Per-round GSI trajectory and missing/present attributes
- Convergence round

---

## Part 3 — Testing Entity Resolution

The retriever handles any phrasing, case, or partial name:

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from src.kg.neo4j_client import Neo4jClient
from src.kg.retriever import KGRetriever

with Neo4jClient() as client:
    r = KGRetriever(client)
    prompts = [
        'generate YAMA the death god',
        'show me the two headed bird from indian mythology',
        'a saola standing in forest',
        'lord shiva with his trident',
        'the bleeding tooth mushroom',
        'ZULU ISICHOLO headdress',
    ]
    for p in prompts:
        ctx = r.retrieve(p)
        print(f'{p!r:55s} → {[e[\"name\"] for e in ctx.primary_entities]}')
"
```

---

## Part 4 — Prompt Synthesis Test

Verify the full KG retrieval + contrastive CoT prompt synthesis:

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from src.kg.neo4j_client import Neo4jClient
from src.kg.retriever import KGRetriever
from src.generation.prompt_synth import PromptSynthesizer

with Neo4jClient() as client:
    retriever = KGRetriever(client)
    synth     = PromptSynthesizer()

    ctx    = retriever.retrieve('Hindu god Yama seated on a water buffalo')
    result = synth.synthesize(ctx)

    print('ORIGINAL :', result.original)
    print()
    print('ENRICHED :', result.enriched)
    print()
    print('CONSTRAINTS:', result.contrastive_cues)
"
```

---

## Key Hyperparameters (Paper Defaults)

| Parameter | Value | Description |
|---|---|---|
| `tau` | 0.85 | GSI convergence threshold |
| `max_k` | 3 | Max SRD iterations |
| `d0` | 0.9 | Initial decay |
| `k_hops` | 1 | KG retrieval hops |
| `guidance_scale` | 15–30 | T2I guidance scale (paper range) |

---

## Troubleshooting

**Neo4j: ServiceUnavailable**
AuraDB free tier pauses after 3 days. Resume at console.neo4j.io.

**OpenAI: RateLimitError during extraction**
Increase sleep between calls:
```bash
python scripts/build_kg.py --domain biology --sleep 2.0
```

**Flux / Janus-Pro: Out of memory**
Enable CPU offloading by editing `src/generation/backbone.py` and adding
`self.pipe.enable_model_cpu_offload()` after loading.

**Entity not found in KG**
The retriever uses three-tier matching including LLM semantic resolution.
If still failing, check that the domain was loaded:
```cypher
MATCH (e:Entity) WHERE e.domain = 'your_domain'
RETURN e.name LIMIT 20
```
