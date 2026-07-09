from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_chroma import Chroma
import json
import os
import requests
from typing import List, Dict, Optional
from Bio import Entrez
from pathlib import Path
from dotenv import load_dotenv

from adapters.chroma import build_ollama_embeddings
from aliases import DISEASE_ALIASES, SPECIES_ALIASES, SPECIMEN_ALIASES

load_dotenv()

# Set default email for NCBI Entrez (required by NCBI API)
Entrez.email = os.getenv("NCBI_EMAIL", "glycan.curator@example.com")
Entrez.api_key = os.getenv("NCBI_API_KEY", None)


# ===========================
# Initialize
# ===========================
embeddings = build_ollama_embeddings()

WORKSPACE_DIR = Path(__file__).parents[2] / "data" / "workspace"
DOID_DATA_PATH = WORKSPACE_DIR / "disease_ontology_v2.txt"
DOID_PERSIST_DIR = WORKSPACE_DIR / "doid_vectorstore"
DOID_COLLECTION = "disease_ontology"

UBERON_DATA_PATH = WORKSPACE_DIR / "uberon_terms.txt"
UBERON_PERSIST_DIR = WORKSPACE_DIR / "uberon_vectorstore"
UBERON_COLLECTION = "uberon_ontology"

GSD_DATA_PATH = WORKSPACE_DIR / "gsd_terms.txt"
GSD_PERSIST_DIR = WORKSPACE_DIR / "gsd_vectorstore"
GSD_COLLECTION = "glycan_structure_dictionary"

# ===========================
# Disease Ontology (DOID) Setup
# ===========================

# Load DOID data and build index
_doid_exact_match_dict: Dict[str, str] = {}

def _init_doid_exact_match_dict():
    """Initialize DOID exact match dictionary for quick lookups."""
    with open(DOID_DATA_PATH, 'r') as file:
                for line_number, line in enumerate(file, 1):
                    # Extract the disease name and the doid at the starting of the line. E.g. "Lidocaine allergy (DOID:0040009) is a drug allergy that has_allergic_trigger lidocaine...."
                    disease_name = line.split(" (DOID:")[0] if " (DOID:" in line else None
                    doid = line.split(" (DOID:")[1].split(") ")[0] if " (DOID:" in line else None
                    if disease_name and doid:
                        _doid_exact_match_dict[disease_name.strip()] = str("DOID:" + doid)

_doid_vectorstore = None

def _init_doid_vectorstore():
    """Initialize DOID vector store lazily on first use."""
    global _doid_vectorstore, _doid_exact_match_dict
    
    if _doid_vectorstore is not None:
        return
    
    if not DOID_DATA_PATH.exists():
        raise FileNotFoundError(f"DOID data file not found: {DOID_DATA_PATH}")
    
    # Load data
    with open(DOID_DATA_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Build exact match dictionary (disease name -> full entry)
    for line in lines:
        if " (DOID:" in line:
            # Extract disease name (before first parenthesis)
            disease_name = line.split(" (DOID:")[0].strip().lower()
            _doid_exact_match_dict[disease_name] = line
            
            # Also extract synonyms if present
            if "exact synonyms:" in line:
                syn_part = line.split("exact synonyms:")[1].split("and related")[0].strip()
                if syn_part and syn_part != "No exact synonyms":
                    for syn in syn_part.split(";"):
                        syn_clean = syn.strip().lower()
                        if syn_clean:
                            _doid_exact_match_dict[syn_clean] = line
    
    # Create documents for vector store
    texts = [Document(page_content=line) for line in lines]
    
    # Check if vector store exists
    DOID_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    if DOID_PERSIST_DIR.exists() and any(DOID_PERSIST_DIR.iterdir()):
        _doid_vectorstore = Chroma(
            persist_directory=str(DOID_PERSIST_DIR),
            collection_name=DOID_COLLECTION,
            embedding_function=embeddings,
        )
        print("Loaded existing DOID vector store.")
    else:
        _doid_vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=str(DOID_PERSIST_DIR),
            collection_name=DOID_COLLECTION,
        )
        print("Created DOID vector store.")

# ===========================
# Uberon Ontology Setup
# ===========================

_uberon_exact_match_dict: Dict[str, str] = {}

def _init_uberon_exact_match_dict():
    """Initialize Uberon exact match dictionary for quick lookups."""
    with open(UBERON_DATA_PATH, 'r') as file:
                for line_number, line in enumerate(file, 1):
                    # Extract the term name and the uberon/cl id at the starting of the line. E.g. "Heart (UBERON:0000948) is a muscular organ...."
                    term_name = line.split(" (UBERON:")[0] if " (UBERON:" in line else None
                    ontology_id = None
                    if " (UBERON:" in line:
                        ontology_id = line.split(" (UBERON:")[1].split(") ")[0]
                    if term_name and ontology_id:
                        _uberon_exact_match_dict[term_name.strip()] = str(ontology_id)

_uberon_vectorstore = None

def _init_uberon_vectorstore():
    """Initialize Uberon vector store lazily on first use."""
    global _uberon_vectorstore, _uberon_exact_match_dict
    
    if _uberon_vectorstore is not None:
        return
    
    if not UBERON_DATA_PATH.exists():
        raise FileNotFoundError(f"Uberon data file not found: {UBERON_DATA_PATH}")
    
    # Load data
    with open(UBERON_DATA_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Build exact match dictionary (term name -> full entry)
    for line in lines:
        if " (UBERON:" in line or " (CL:" in line:
            # Extract term name (before first parenthesis)
            term_name = line.split(" (")[0].strip().lower()
            _uberon_exact_match_dict[term_name] = line
            
            # Extract synonyms
            if "Synonyms:" in line:
                syn_part = line.split("Synonyms:")[1].split(".")[0].strip()
                if syn_part and syn_part.lower() != "none":
                    for syn in syn_part.split(";"):
                        syn_clean = syn.strip().lower()
                        if syn_clean:
                            _uberon_exact_match_dict[syn_clean] = line
    
    # Create documents for vector store
    texts = [Document(page_content=line) for line in lines]
    
    # Check if vector store exists
    UBERON_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    if UBERON_PERSIST_DIR.exists() and any(UBERON_PERSIST_DIR.iterdir()):
        _uberon_vectorstore = Chroma(
            persist_directory=str(UBERON_PERSIST_DIR),
            collection_name=UBERON_COLLECTION,
            embedding_function=embeddings,
        )
        print("Loaded existing Uberon vector store.")
    else:
        _uberon_vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=str(UBERON_PERSIST_DIR),
            collection_name=UBERON_COLLECTION,
        )
        print("Created Uberon vector store.")

# ===========================
# Glycan Structure Dictionary (GSD) Setup
# ===========================

_gsd_exact_match_dict: Dict[str, str] = {}
_gsd_vectorstore = None

def _init_gsd_vectorstore():
    """Initialize GSD vector store lazily on first use."""
    global _gsd_vectorstore, _gsd_exact_match_dict
    
    if _gsd_vectorstore is not None:
        return
    
    if not GSD_DATA_PATH.exists():
        raise FileNotFoundError(f"GSD data file not found: {GSD_DATA_PATH}")
    
    # Load data
    with open(GSD_DATA_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Build exact match dictionary (glycan name -> full entry)
    for line in lines:
        # Extract glycan name (before first pipe |)
        if " | " in line:
            glycan_name = line.split(" | ")[0].strip().lower()
            _gsd_exact_match_dict[glycan_name] = line
            
            # Also extract exact synonyms if present
            if "Exact synonyms:" in line and "Exact synonyms: None" not in line:
                syn_part = line.split("Exact synonyms:")[1].split(" | ")[0].strip()
                if syn_part:
                    for syn in syn_part.split(";"):
                        syn_clean = syn.strip().lower()
                        if syn_clean:
                            _gsd_exact_match_dict[syn_clean] = line
    
    # Create documents for vector store
    texts = [Document(page_content=line) for line in lines]
    
    # Check if vector store exists
    GSD_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    if GSD_PERSIST_DIR.exists() and any(GSD_PERSIST_DIR.iterdir()):
        _gsd_vectorstore = Chroma(
            persist_directory=str(GSD_PERSIST_DIR),
            collection_name=GSD_COLLECTION,
            embedding_function=embeddings,
        )
        print("Loaded existing GSD vector store.")
    else:
        _gsd_vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=str(GSD_PERSIST_DIR),
            collection_name=GSD_COLLECTION,
        )
        print("Created GSD vector store.")

# ===========================
# Tool Functions
# ===========================

@tool
def onto_doid_tool(entities: List[str]) -> str:
    """ 
    This tool searches the Disease Ontology (DOID) to find matching disease concepts
    for the provided list of disease names. It first performs exact string matching
    against known disease names and synonyms ('exact'), then falls back to semantic
    embedding search ('semantic') for terms without exact matches.
    
    Args:
        entities: List of disease term strings to map to DOID identifiers
        Example: {'entities': ['diabetes mellitus', 'Parkinson disease', 'asthma']}
        
    Returns:
        JSON string containing searching results:
        - DOID preferred term labels
        - DOID identifiers
        - Synonyms (if present)
        - Definitions or brief descriptions
        - Others
    """
    if not entities:
        return json.dumps({"results": [], "message": "No entities provided"})

    # Initialize exact match dictionary
    if not _doid_exact_match_dict:
        _init_doid_exact_match_dict()
    
    results = []
    
    for entity in entities:
        entity_lower = entity.strip().lower()

        # Step 0: Check static alias cache
        if entity_lower in DISEASE_ALIASES:
            results.append({
                "query": entity,
                "match_type": "alias",
                "result": DISEASE_ALIASES[entity_lower]
            })
            continue

        # Step 1: Try exact match
        if entity_lower in _doid_exact_match_dict:
            results.append({
                "query": entity,
                "match_type": "exact",
                "result": _doid_exact_match_dict[entity]
            })
            continue

        # Step 2: Fall back to embedding search (lazy init vectorstore)
        try:
            _init_doid_vectorstore()
            retriever = _doid_vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            docs = retriever.invoke(entity)
            
            if docs:
                matches = []
                for i, doc in enumerate(docs):
                    matches.append({
                        "rank": i + 1,
                        "entry": doc.page_content
                    })
                
                results.append({
                    "query": entity,
                    "match_type": "semantic",
                    "results": matches
                })
            else:
                results.append({
                    "query": entity,
                    "match_type": "none",
                    "message": "No matches found"
                })
        except Exception as e:
            results.append({
                "query": entity,
                "match_type": "error",
                "message": str(e)
            })
    
    return json.dumps({"results": results}, indent=2)


@tool
def onto_gsd_tool(entities: List[str]) -> str:
    """ 
    This tool searches the Glycan Structure Dictionary (GSD) to find matching glycan structures
    for the provided list of glycan names. It first performs exact string matching against known
    glycan names and synonyms ('exact'), then falls back to semantic embedding search ('semantic').
    
    Args:
        entities: List of glycan term strings to map to GSD identifiers
        Example: {'entities': ['sialyl Lewis x', 'lactose', 'N-glycan', 'ganglioside GM1']}
        
    Returns:
        JSON string containing searching results:
        - GSD preferred term labels
        - GSD identifiers (standardized UUIDs)
        - Exact synonyms (alternative names for the glycan)
        - Descriptions (detailed biological context)
        - Others
    """
    _init_gsd_vectorstore()
    
    if not entities:
        return json.dumps({"results": [], "message": "No entities provided"})
    
    results = []
    
    for entity in entities:
        entity_lower = entity.strip().lower()
        
        # Step 1: Try exact match first
        if entity_lower in _gsd_exact_match_dict:
            results.append({
                "query": entity,
                "match_type": "exact",
                "result": _gsd_exact_match_dict[entity_lower]
            })
            continue
        
        # Step 2: Fall back to embedding search
        try:
            retriever = _gsd_vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            docs = retriever.invoke(entity)
            
            if docs:
                matches = []
                for i, doc in enumerate(docs):
                    matches.append({
                        "rank": i + 1,
                        "entry": doc.page_content
                    })
                
                results.append({
                    "query": entity,
                    "match_type": "semantic",
                    "results": matches
                })
            else:
                results.append({
                    "query": entity,
                    "match_type": "none",
                    "message": "No matches found"
                })
        except Exception as e:
            results.append({
                "query": entity,
                "match_type": "error",
                "message": str(e)
            })
    
    return json.dumps({"results": results}, indent=2)


@tool
def onto_uberon_tool(entities: List[str]) -> str:
    """ 
    This tool searches Uberon/Cell Ontology (UBERON/CL) to find matching anatomical
    structures or cell types for the provided list of terms. It first performs exact
    string matching against known term names and synonyms ('exact'), then falls back
    to semantic embedding search ('semantic') for terms without exact matches.
    
    Args:
        entities: List of anatomical or cell type term strings to map to UBERON/CL IDs
        Example: {'entities': ['heart', 'hepatocyte', 'cerebral cortex']}
        
    Returns:
        JSON string containing searching results:
        - UBERON/CL preferred term labels
        - UBERON/CL identifiers
        - Synonyms (if present)
        - Definitions or brief descriptions
        - Others
    """
    _init_uberon_vectorstore()
    
    if not entities:
        return json.dumps({"results": [], "message": "No entities provided"})
    
    results = []
    
    for entity in entities:
        entity_lower = entity.strip().lower()

        # Step 0: Check static alias cache
        if entity_lower in SPECIMEN_ALIASES:
            results.append({
                "query": entity,
                "match_type": "alias",
                "result": SPECIMEN_ALIASES[entity_lower]
            })
            continue

        # Step 1: Try exact match
        if entity_lower in _uberon_exact_match_dict:
            results.append({
                "query": entity,
                "match_type": "exact",
                "result": _uberon_exact_match_dict[entity]
            })
            continue

        # Step 2: Fall back to embedding search
        try:
            retriever = _uberon_vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            docs = retriever.invoke(entity)
            
            if docs:
                matches = []
                for i, doc in enumerate(docs):
                    matches.append({
                        "rank": i + 1,
                        "entry": doc.page_content
                    })
                
                results.append({
                    "query": entity,
                    "match_type": "semantic",
                    "results": matches
                })
            else:
                results.append({
                    "query": entity,
                    "match_type": "none",
                    "message": "No matches found"
                })
        except Exception as e:
            results.append({
                "query": entity,
                "match_type": "error",
                "message": str(e)
            })
    
    return json.dumps({"results": results}, indent=2)


@tool
def onto_cellline_tool(entities: List[str]) -> str:
    """ 
    This tool searches the Cellosaurus API to find matching cell line entries for the
    provided list of cell line names. It returns standardized CVCL identifiers along
    with primary names and synonyms for each query.
    
    Args:
        entities: List of cell line name strings to search for
        Example: {'entities': ['HEK293', 'HeLa', 'CHO-K1']}
        
    Returns:
        Formatted string containing searching results:
        - Cellosaurus CVCL identifiers
        - Primary identifiers (preferred names)
        - Synonyms (up to 10 shown, with count if more exist)
        - Others
    """
    
    if not entities:
        return "Error: No cell line names provided"
    
    output_str = ""
    
    for cell_line_name in entities:
        query_name = cell_line_name.strip()
        output_str += f"Cell line: {query_name}\n"
        
        try:
            # Query Cellosaurus API - use f-string to avoid double URL encoding
            url = f"https://api.cellosaurus.org/search/cell-line?q=id%3A{query_name}&start=0&rows=10&format=json&fld=id&fld=sy&fld=idsy&fld=ac"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            cell_lines = data.get('Cellosaurus', {}).get('cell-line-list', [])
            
            if not cell_lines:
                output_str += "  (No matches found)\n\n"
                continue
            
            # Process each matching cell line
            for cell_line in cell_lines:
                # Get primary accession (CVCL_XXXX)
                accessions = cell_line.get('accession-list', [])
                primary_acc = None
                for acc in accessions:
                    if acc.get('type') == 'primary':
                        primary_acc = acc.get('value')
                        break
                
                # Get identifier (primary name) and synonyms
                names = cell_line.get('name-list', [])
                identifier = None
                synonyms = []
                
                for name_entry in names:
                    name_type = name_entry.get('type')
                    name_value = name_entry.get('value')
                    
                    if name_type == 'identifier' and not identifier:
                        identifier = name_value
                    elif name_type == 'synonym' and name_value not in synonyms:
                        synonyms.append(name_value)
                
                # Format output line
                if primary_acc:
                    output_str += f"  {primary_acc}"
                    if identifier:
                        output_str += f" | Primary: {identifier}"
                    if synonyms:
                        # Limit synonyms to first 10 to avoid excessive output
                        syn_list = synonyms[:10]
                        output_str += f" | Synonyms: {', '.join(syn_list)}"
                        if len(synonyms) > 10:
                            output_str += f" ... ({len(synonyms) - 10} more)"
                    output_str += "\n"
            
            output_str += "\n"
            
        except requests.exceptions.RequestException as e:
            output_str += f"  Error fetching data: {str(e)}\n\n"
        except Exception as e:
            output_str += f"  Error processing data: {str(e)}\n\n"
    
    return output_str


@tool
def onto_protein_tool(protein_names: List[str], species_name: Optional[str] = None) -> str:
    """
    This tool searches UniProt for matching protein entries given protein names and an
    optional species name. It returns a compact TSV with canonical accessions and gene
    symbols, limited to the top hits per query.

    Args:
        protein_names: List of protein name strings to search for
        species_name: Optional species name string (e.g., 'human', 'Homo sapiens', 'mouse')
        Example: {'protein_names': ['insulin', 'IgG'], 'species_name': 'human'}

    Returns:
        TSV-formatted string containing searching results:
        - UniProt accession IDs
        - Protein names (preferred and synonyms)
        - Primary gene symbols
        - HGNC cross-references (when available)
        - Others
    """
    import requests

    if not protein_names:
        return "Error: No protein names provided"

    output_str = ""

    for name in protein_names:
        query_name = name.strip()
        query = f'protein_name:{query_name} AND organism_name:{species_name}' if species_name else f'protein_name:{query_name}'
        try:
            response = requests.get(
                f'https://rest.uniprot.org/uniprotkb/search',
                params={
                    'query': query,
                    'fields': 'accession,protein_name,gene_primary,xref_hgnc',
                    'format': 'tsv'
                },
                timeout=10
            )
            response.raise_for_status()
            
            # Get first 9 lines (header + 8 results)
            response_lines = response.text.splitlines()[:9]
            response_str = "\n".join(response_lines)
            
            output_str += f"Matches for {query_name}:\n{response_str}\n\n"
            
        except requests.exceptions.RequestException as e:
            output_str += f"Matches for {query_name}:\nError fetching data: {str(e)}\n\n"
        except Exception as e:
            output_str += f"Matches for {query_name}:\nError processing data: {str(e)}\n\n"
    
    return output_str


@tool
def onto_taxonomy_tool(species_names: List[str]) -> str:
    """ 
    This tool searches the NCBI Taxonomy database to find organism taxonomy records for
    the provided list of species names (scientific names, common names, or synonyms).
    
    Args:
        species_names: List of species name strings to search for
        Example: {'species_names': ['Homo sapiens', 'mouse', 'E. coli']}
        
    Returns:
        TSV-formatted string containing searching results:
        - Taxonomy IDs
        - Scientific names
        - Common names
        - Included or related names (if present)
        - Others
    """
    from Bio import Entrez
    
    # Set email for NCBI Entrez (required by NCBI)
    if not Entrez.email:
        Entrez.email = "glycan.curator@example.com"
    
    if not species_names:
        return "Error: No species names provided"
    
    output_str = ""
    
    for name in species_names:
        query_name = name.strip()
        output_str += f"{query_name}:\ntax_id\tscientific_name\tcommon_name\tincludes\tgenbank_common_name\n"

        if query_name.lower() in SPECIES_ALIASES:
            output_str += SPECIES_ALIASES[query_name.lower()] + "\n\n"
            continue
        
        try:
            # Search for taxonomy IDs
            search_handle = Entrez.esearch(db="taxonomy", term=query_name, retmode="xml")
            search_result = Entrez.read(search_handle)
            search_handle.close()
            
            tax_ids = search_result.get('IdList', [])
            
            if not tax_ids:
                output_str += "(No matches found)\n\n"
                continue
            
            # Fetch detailed information for found IDs
            fetch_handle = Entrez.efetch(db="taxonomy", id=tax_ids, retmode="xml")
            taxonomy_records = Entrez.read(fetch_handle)
            fetch_handle.close()
            
            for record in taxonomy_records:
                tax_id = record.get('TaxId', 'N/A')
                scientific_name = record.get('ScientificName', 'N/A')
                
                # Extract other names
                other_names = record.get('OtherNames', {})
                common_names = other_names.get('CommonName', [])
                common_name_str = "(None)" if not common_names else ", ".join(common_names)
                
                includes = other_names.get('Includes', [])
                includes_str = "(None)" if not includes else ", ".join(includes)
                
                genbank_name = other_names.get('GenbankCommonName', '(None)')
                
                output_str += f"{tax_id}\t{scientific_name}\t{common_name_str}\t{includes_str}\t{genbank_name}\n"
            
            output_str += "\n"
            
        except Exception as e:
            output_str += f"Error fetching taxonomy data: {str(e)}\n\n"
    
    return output_str
