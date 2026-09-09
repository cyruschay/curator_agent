<a id="readme-top"></a>

<!--
*** This doc is created using the Best-README-Template style:
*** https://github.com/othneildrew/Best-README-Template/
-->

<!-- PROJECT SHIELDS -->
[![Pipeline][pipeline-shield]][pipeline-url]
[![Issues][issues-shield]][issues-url]

[![Ollama][ollama-shield]][ollama-url]
[![LangGraph][langgraph-shield]][langgraph-url]
[![Chroma][chroma-shield]][chroma-url]

<br>

<div align="center">
	<h1 align="center">Glycan Biomarker Curator Agent</h1>
	<p align="center">
		LLM-assisted pipeline for screening glycoscience literature and curating glycan-disease biomarker relations.
		<br />
		<a href="#about-this-project"><strong>Explore the docs >></strong></a>
		<br />
		<br />
		<a href="#usage">Usage</a>
		&middot;
		<a href="#data">Data</a>
	</p>
</div>

<div align="left">
	<div style="display: inline-block; text-align: left; border: 1px solid #888; border-radius: 30px; max-width: 340px;">
		<details style="padding: 10px 20px 0px;">
			<summary><strong>&nbsp&nbspTable of Contents</strong></summary>
			<ol>
				<li><a href="#about-this-project">About this project</a></li>
				<li><a href="#getting-started">Getting started</a></li>
				<li><a href="#usage">Usage</a></li>
				<li><a href="#data">Data source</a></li>
				<li><a href="#data-model">Data model</a></li>
			</ol>
		</details>
	</div>
</div>

## About This Project

This repository provides an end-to-end curation workflow for glycan biomarker literature. It reads PubMed abstract records, screens for biomarker relevance, performs glycan entity and relation extraction, maps entities to biomedical ontologies, validates extracted claims against evidence text, scores confidence, and exports structured JSONL curation outputs.

The workflow is implemented as a LangGraph state machine in `src/agent/graph.py` with node-level responsibilities for:

- abstract screening (`N01`)
- optional full-text retrieval from PMC (`N02`)
- glycan NER (`N03/N03a`)
- role-driven relation extraction (`N04/N04a`): identify biomarker role(s), then run
  role-specific describe + structure prompts/schemas per role
- deterministic ontology mapping (`N05`): exact/alias → semantic → bounded adjudication
- role-aware validation/refinement and deduplication (`N06`)
- batch JSONL export (`N08`)

Core capabilities include:

- Local Ollama inference for reasoning and extraction models.
- Hybrid ontology mapping using exact string lookup plus Chroma semantic retrieval.
- Resume-safe curation runs via `curated_list.jsonl` hash-based tracking.
- Structured, auditable outputs (curations, runlogs, violations, tool calls, model metadata).

<p align="right"><a href="#readme-top">back to top ^</a></p>

## Getting Started

### Prerequisites

- Linux or HPC environment with Python `3.12`
- Ollama installed locally
- (Optional, recommended) `NCBI_API_KEY` environment variable for higher Entrez API rate limits

Install Ollama:

```sh
curl -fsSL https://ollama.com/install.sh | sh
```

Or use the official installer: https://ollama.com/download

### Installation

1. Enter this repository:

```sh
cd curator_agent
```

2. Pull required Ollama models.

This project uses:

- reasoning model: `gpt-oss:20b`
- embedding model: `bge-large:latest`

```sh
ollama pull gpt-oss:20b
ollama pull bge-large:latest
ollama list
```

3. Create and activate a Python environment:

```sh
conda create --name gly_env python=3.12
conda activate gly_env
```

4. Install Python dependencies:

```sh
python -m pip install -r requirements.txt
```

5. Configure runtime settings in `config.yaml`.

Important fields:

- `ollama_models`: model names
- `ollama_params`: generation parameters
- `paths.input_abstracts`: input JSONL filename under `data/raw/abstracts/`
- `paths.curation_output_dir`: output subdirectory under `data/processed/curation/`

### Run Modes

Local workstation run:

```sh
bash main_pc.sh
```

This script starts `ollama serve` and runs:

```sh
python -u src/agent/graph.py
```

Slurm/HPC run:

```sh
sbatch main_slurm.sh
```

The Slurm script loads an Ollama module, starts the Ollama server, and executes the same graph pipeline.

<p align="right"><a href="#readme-top">back to top ^</a></p>

## Usage

### Project Structure

```bash
.
|-- README.md
|-- config.yaml                        # Model + path configuration
|-- main_pc.sh                         # Local launcher
|-- main_slurm.sh                      # Slurm launcher
|-- requirements.txt
|-- data
|   |-- raw
|   |   `-- abstracts                  # PubMed abstract JSONL inputs
|   `-- processed
|       `-- curation                   # Batched curation outputs
|-- logs                               # Slurm stdout/stderr
|-- scripts
|   |-- fetch_abstracts.py             # PubMed query + abstract ingestion
|   `-- 04_utils                       # XML/fulltext helper scripts
|-- src
|   `-- agent
|       |-- graph.py                   # LangGraph pipeline (N01-N08)
|       |-- prompts.py                 # System prompts + per-role prompt builders
|       |-- roles.py                   # Per-biomarker-role curation specs (from biomarker.md)
|       |-- schemas.py                 # Input schema + per-role structure schemas
|       |-- states.py                  # AgentState typed model
|       |-- ontology.py                # Deterministic ontology resolvers (exact/alias/semantic)
|       |-- utils.py                   # Retrieval/indexing/normalization helpers
|       `-- vectorstores               # Local ontology resources/chroma stores
`-- tests
		|-- integration_tests
		`-- unit_tests
```

### LLM Workflow

The graph in `src/agent/graph.py` runs the following workflow:

1. `N01_AbstractScreener`
Decides `RELEVANT/IRRELEVANT`, study type, and whether full text is needed.

2. `N02_FullTextRetrieval` (conditional)
If PMCID is available and full text is needed, fetches and parses PMC JATS XML.

3. `N03/N03a` Glycan NER
Extracts glycan structure terms and sentence evidence from full text or abstract mode.
Excludes umbrella terms and undecipherable peak labels (e.g., `GP20`, `IGP33`), and keeps the
glycan term pure (no protein/tissue names; bare adjectives become `<adjective> glycan`).

4. `N04/N04a` Relation Extraction (role-driven)
Three simple steps so each call stays within the small model's limits:
- role identification: which of the 7 BEST biomarker roles the article supports, and the glycans for each;
- per-role *describe*: role-specific guided questions + definition + the role's required/optional fields;
- per-role *structure*: grammar-constrained JSON with that role's schema.
Roles are a non-exclusive list, so a glycan playing several roles yields one relation with a role
list and role-keyed annotations. Role specs live in `src/agent/roles.py` (from `biomarker.md`).

5. `N05_OntologyMapper` (deterministic)
No LLM tool-calling loop. Each entity is resolved by exact/alias lookup, then semantic search
(top-k > 5); only semantic-only hits go to a single bounded adjudication call. Type-aware routing:
- glycan → GSD (exact + semantic only; **no fuzzy match**)
- disease → DOID
- specimen → UBERON/CL (tissue/organ/fluid/circulating cell) or Cellosaurus (cell line)
- species → NCBI Taxonomy
- protein → UniProt (single gene) or Complex Portal (complex)
A match is never forced; entities with no true equivalent stay unmapped.

6. `N06_ValidateRefine`
Role-aware validation: checks the evidence supports the assigned role and its required fields,
applies fix/split/reject, and deduplicates relations.

7. `N08_Exporter`
Writes batched JSONL artifacts to `data/processed/curation/<run_name>/`.

### Input Preparation

To fetch abstracts from PubMed and generate a JSONL queue:

```sh
python scripts/fetch_abstracts.py
```

Outputs are written under `data/raw/abstracts/` (PMIDs, query translation, and abstracts JSONL).

<p align="right"><a href="#readme-top">back to top ^</a></p>

## Data

### Data Source

| Resource | Role in Pipeline | Access Method |
| --- | --- | --- |
| PubMed | Primary literature discovery + abstracts | NCBI Entrez via `scripts/fetch_abstracts.py` |
| PubMed Central (PMC) | Full-text retrieval for selected PMCID articles | Entrez `efetch(db="pmc")` in `utils.retrieve_pmc_fulltext` |
| Glycan Structure Dictionary (GSD) | Glycan term mapping target | Local text resource + Chroma in `src/agent/vectorstores/` |
| Disease Ontology (DOID) | Disease normalization | Local text resource + Chroma |
| Uberon / CL | Specimen and anatomy normalization | Local text resource + Chroma |
| Cellosaurus | Cell line normalization | REST API lookup |
| NCBI Taxonomy | Species normalization | Entrez taxonomy queries |
| UniProt | Protein normalization | UniProt REST queries |

### Output Data Artifacts

Each run writes batch files such as:

- `curated_list.jsonl` (resume and status tracking)
- `curations_batch_001.jsonl` (primary curation payload)
- `runlog_batch_001.jsonl` (counts and run summary)
- `violations_batch_001.jsonl` (validation issues)
- `errors_batch_001.jsonl` (error messages)
- `reasonings_batch_001.jsonl` (model reasoning traces)
- `tool_calls_batch_001.jsonl` (tool invocation records)
- `metadata_batch_001.jsonl` (LLM metadata)

<p align="right"><a href="#readme-top">back to top ^</a></p>

## Data Model

### Input Record Schema (JSONL)

Each input line must satisfy `src/agent/schemas.py` and includes fields such as:

- `processing_id`, `pmid`, `pmcid`
- `title`, `abstract`
- optional metadata: keywords, MeSH, chemicals, journal, article date

Example:

```json
{
	"processing_id": "PID:0000001",
	"pmid": "40662295",
	"pmcid": "12269659",
	"title": "Trispecific SEED antibodies engineered for neutrophil-mediated cell killing.",
	"abstract": "Immunoglobulin (Ig) A has attracted interest...",
	"keywords": ["ADCC", "EGFR"],
	"mesh_names": ["Humans", "Protein Engineering"]
}
```

### Core Export Object (`curations_batch_XXX.jsonl`)

Each exported line contains:

- article identity (`pid`, `pmid`, `pmcid`, `title`)
- screening decision and confidence
- entity list (glycan structure term candidates)
- relation list with mapped identifiers and evidence
- provenance (ingest hash, batch offsets)

Relation objects include:

- glycan fields: `glycan_name`, `glycan_mapped_name`, `glycan_id`
- disease fields: `disease_name`, `disease_mapped_name`, `disease_id`
- context: `specimen`, `species_name/species_id`, `protein_name/protein_id`
- role: `biomarker_role` (list), `role_annotations` (role-keyed required/optional fields),
  `is_multicomponent`
- extraction metadata: `direction`, `metrics`, `method_names`, `negated_or_hedged`
- evidence payload: sentence text/index/section

Example relation snippet:

```json
{
	"glycan_name": "core fucosylation",
	"glycan_mapped_name": "core fucosylated N-glycan",
	"glycan_id": "GSD:...",
	"disease_name": "hepatocellular carcinoma",
	"disease_mapped_name": "hepatocellular carcinoma",
	"disease_id": "DOID:684",
	"specimen": {
		"original": "serum",
		"mapped_name": "blood serum",
		"mapped_id": "UBERON:0001977",
		"category": "tissue",
		"ontology": "UBERON"
	},
	"biomarker_role": ["diagnostic", "monitoring"],
	"role_annotations": {
		"diagnostic": {"target_condition": "hepatocellular carcinoma", "diagnostic_task": "presence_absence", "comparison_group": "healthy controls"},
		"monitoring": {"repeated_measurement": "serial serum sampling", "monitored_entity": "tumor burden", "change_direction": "rising"}
	},
	"is_multicomponent": false,
	"direction": "increased",
	"metrics": [{"name": "AUC", "value": 0.84, "raw": "AUC 0.84"}],
	"evidence_sentences": [
		{"sentence": "...", "section": "ABSTRACT", "sentence_index": 7}
	]
}
```

<p align="right"><a href="#readme-top">back to top ^</a></p>

<!-- MARKDOWN LINKS -->

[pipeline-shield]: https://img.shields.io/badge/pipeline-langgraph-teal?style=for-the-badge
[pipeline-url]: https://github.com/glygener
[issues-shield]: https://img.shields.io/badge/issues-open-mediumaquamarine?style=for-the-badge
[issues-url]: https://github.com/glygener
[ollama-shield]: https://img.shields.io/badge/ollama-local%20llm-008080?style=for-the-badge
[ollama-url]: https://ollama.com
[langgraph-shield]: https://img.shields.io/badge/langgraph-workflow-008080?style=for-the-badge
[langgraph-url]: https://www.langchain.com/langgraph
[chroma-shield]: https://img.shields.io/badge/chroma-vector%20store-008080?style=for-the-badge
[chroma-url]: https://www.trychroma.com
