# Dependencies: Biopython
# Note: Set your NCBI API key in the environment variable NCBI_API_KEY for higher rate limits.

from Bio import Entrez
from datetime import datetime
import time
import os
import json
from pathlib import Path
from tqdm import tqdm

# ───────────────────────────────────────────────────────────────────────────────
# User-defined parameters
term = "(('glycomics'[MeSH Terms] OR 'glycosylation'[MeSH Terms] OR 'glycan'[Title/Abstract] OR 'glycans'[Title/Abstract]) AND 'biomarkers'[MeSH Terms]) NOT (Review[Publication Type])"
start_year = 1900
end_year = "current"             # Use "current" to get the current year
Entrez.email = "your.email@here.com"  # Replace with your email address
NCBI_API_KEY = "YOUR_API_KEY"    # Optional: replace with your key or set env var
OUTPUT_NAME = "pmids.txt"
BATCH_SIZE = 50                  # PMIDs fetched per batch for abstracts
# ───────────────────────────────────────────────────────────────────────────────

SRC_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = SRC_DIR / "data"
ABSTRACTS_DIR = SRC_DIR / "data/raw/abstracts"
ABSTRACTS_DIR.mkdir(parents=True, exist_ok=True)

TRANSLATION_FILE = ABSTRACTS_DIR / "translation.txt"
PMIDS_FILE = ABSTRACTS_DIR / OUTPUT_NAME
ABSTRACTS_JSONL = ABSTRACTS_DIR / "abstracts.jsonl"

API_LIMIT = 9999  # You just cannot fetch more than 9,999 PMIDs


def check_api_key():
    env_key = os.getenv("NCBI_API_KEY")
    if env_key and env_key != "YOUR_API_KEY":
        Entrez.api_key = env_key
        print("[Message] Environmental variable NCBI_API_KEY detected and using.")
    else:
        print("[Message] No NCBI_API_KEY detected. Using default rate limits.")
        print("          Consider setting NCBI_API_KEY for higher rate limits.")
        print("          See https://support.nlm.nih.gov/kbArticle/?pn=KA-05317 for more information.")

def count_pmids(term, start_year, end_year):
    print(f"{start_year}-{end_year}", end=" ", flush=True)
    handle = Entrez.esearch(
        db="pubmed", term=term,
        mindate=f"{start_year}/01/01",
        maxdate=f"{end_year}/12/31",
        retmax=0
    )
    record = Entrez.read(handle)
    handle.close()
    count = int(record["Count"])
    if (end_year - start_year == 1) and count > API_LIMIT:
        print(f"[Warning] There are {API_LIMIT} hits in {start_year} - {end_year}. Not all will be retrieved.")
    return count

def fetch_translation(term, start_year, end_year):
    handle = Entrez.esearch(
        db="pubmed", term=term,
        mindate=f"{start_year}/01/01",
        maxdate=f"{end_year}/12/31",
        retmax=0
    )
    record = Entrez.read(handle)
    handle.close()
    return record['QueryTranslation']

def fetch_pmids(term, start_year, end_year, batch):
    handle = Entrez.esearch(
        db="pubmed", term=term,
        mindate=f"{start_year}/01/01",
        maxdate=f"{end_year}/12/31",
        retmax=batch,
        sort="pub_date"
    )
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]

def batch_retrieval_by_recursive_splitting(term, start_year, end_year):
    queue = [(start_year, end_year)]
    all_pmids = []

    while queue:
        a, b = queue.pop()
        n = count_pmids(term, a, b)
        if n == 0:
            continue  # nothing in this window
        if n <= API_LIMIT or a == b:  # leaf – safe to fetch
            all_pmids.extend(fetch_pmids(term, a, b, API_LIMIT))
            time.sleep(0.3)
        else:  # split the denser half
            mid = (a + b) // 2
            queue.append((a, mid))
            queue.append((mid + 1, b))
    return all_pmids

def fetch_xml(pmids):
    handle = Entrez.efetch(db="pubmed", id=",".join(pmids), retmode="xml")
    xml = Entrez.read(handle)
    handle.close()
    return xml

def extract_fields(article):
    def pmid_to_pmcid(pmid: str):
        h = Entrez.elink(dbfrom="pubmed", id=pmid, linkname="pubmed_pmc")
        rec = Entrez.read(h)
        for lset in rec:
            for ldb in lset.get("LinkSetDb", []):
                if ldb.get("LinkName") == "pubmed_pmc":
                    for link in ldb.get("Link", []):
                        return link["Id"]
        return None

    medline_citation = article.get("MedlineCitation", {})
    pmid = medline_citation.get("PMID", "")
    pmcid = pmid_to_pmcid(pmid)
    journal_abbr = medline_citation.get("Article", {}).get("Journal", {}).get("ISOAbbreviation", "")
    article_date = medline_citation.get("Article", {}).get("ArticleDate", [])

    article_data = medline_citation.get("Article", {})
    title = article_data.get("ArticleTitle", "")

    # Abstract Text
    abstract_field = article_data.get("Abstract", {}).get("AbstractText", "")
    if isinstance(abstract_field, list):
        abstract = " ".join([str(x) for x in abstract_field])
    else:
        abstract = str(abstract_field)

    # Keywords (may be nested)
    keyword_list = []
    if "KeywordList" in medline_citation and medline_citation["KeywordList"]:
        kw_field = medline_citation["KeywordList"]
        if isinstance(kw_field[0], list):
            keywords = kw_field[0]
        else:
            keywords = kw_field
        keyword_list = [str(k) for k in keywords] if keywords else []

    # Chemical substances
    chemical_names, chemical_rn, chemical_uid = [], [], []
    for chem in medline_citation.get("ChemicalList", []):
        name = str(chem.get("NameOfSubstance", ""))
        rn = str(chem.get("RegistryNumber", ""))
        ui = ""
        try:
            ui = chem["NameOfSubstance"].attributes.get("UI", "")
        except Exception:
            ui = ""
        if name:
            chemical_names.append(name)
        if rn:
            chemical_rn.append(rn)
        if ui:
            chemical_uid.append(ui)

    # MeSH
    mesh_names, mesh_uid = [], []
    for mesh in medline_citation.get("MeshHeadingList", []):
        dname = str(mesh.get("DescriptorName", ""))
        dui = ""
        try:
            dui = mesh["DescriptorName"].attributes.get("UI", "")
        except Exception:
            dui = ""
        if dname:
            mesh_names.append(dname)
        if dui:
            mesh_uid.append(dui)

    clean_html_tags = lambda text: text.replace("<i>", "").replace("</i>", "").replace("<b>", "").replace("</b>", "").replace("<u>", "").replace("</u>", "").replace("\"", "'").strip()

    return {
        "pmid": pmid if pmid else "[ERROR]",
        "pmcid": pmcid if pmcid else None,
        "title": clean_html_tags(title) if title else None,
        "abstract": clean_html_tags(abstract) if abstract else None,
        "keywords": keyword_list if keyword_list else [],
        
        "chemical_names": chemical_names if chemical_names else [],
        "chemical_rn": chemical_rn if chemical_rn else [],
        "chemical_uid": chemical_uid if chemical_uid else [],
        
        "mesh_names": mesh_names if mesh_names else [],
        "mesh_uid": mesh_uid if mesh_uid else [],
        
        "journal_abbr": journal_abbr,
        "article_date": article_date if article_date else []
    }


if __name__ == "__main__":
    # Resolve end year
    if end_year == "current":
        end_year = datetime.now().year

    # Configure Entrez
    Entrez.email = Entrez.email
    check_api_key()

    # (1) Query translation
    print("[Message] (1) Getting the query translation of the search term...")
    translation = fetch_translation(term, start_year, end_year)
    TRANSLATION_FILE.write_text(translation + "\n", encoding="utf-8")
    print(f"[Message] Saved translation to {TRANSLATION_FILE.parent.name}/{TRANSLATION_FILE.name}")

    # (2) PMID retrieval via recursive splitting
    print("[Message] (2) Fetching PMIDs...")
    print("[Message] Recursive splitting year range: ", end=" ", flush=True)
    pmids = batch_retrieval_by_recursive_splitting(term, start_year, end_year)
    PMIDS_FILE.write_text("\n".join(pmids) + "\n", encoding="utf-8")
    print(f"\n[Message] Saved {len(pmids):,} PMIDs to {PMIDS_FILE.parent.name}/{PMIDS_FILE.name}")

    # (3) Fetch abstracts and write jsonl
    print("[Message] (3) Fetching abstracts and writing JSONL...")
    row = 1  # Processing index
    with ABSTRACTS_JSONL.open("w", encoding="utf-8") as jf:
        for i in tqdm(range(0, len(pmids), BATCH_SIZE),
                      desc="Processing", unit="batch", colour="green"):
            batch_pmids = pmids[i:i + BATCH_SIZE]
            xml = fetch_xml(batch_pmids)
            for article in xml.get("PubmedArticle", []):
                rec = extract_fields(article)
                rec_out = {"processing_id": "PID:" + str(row).zfill(7)} | rec
                jf.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                row += 1

    print(f"[Complete] Abstracts written to JSONL:\n  {ABSTRACTS_JSONL.parent.name}/{ABSTRACTS_JSONL.name}")
