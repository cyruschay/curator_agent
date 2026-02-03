from Bio import Entrez
import os, re, pathlib, requests
Entrez.email = "you@example.com"
if os.getenv("NCBI_API_KEY"):
    Entrez.api_key = os.getenv("NCBI_API_KEY")

# 1) From a PMID: find repository accessions via ELink
def get_repository_links_from_pubmed(pmid: str):
    link_dbs = [
        "pubmed_geo","pubmed_sra","pubmed_bioproject","pubmed_biosample",
        "pubmed_gds","pubmed_pdb","pubmed_clinicaltrials","pubmed_databank"
    ]
    out = {}
    for db in link_dbs:
        h = Entrez.elink(dbfrom="pubmed", id=pmid, linkname=db)
        rec = Entrez.read(h)
        links = []
        for lset in rec:
            for link in lset.get("LinkSetDb", []):
                for item in link.get("Link", []):
                    links.append(item["Id"])
        if links:
            out[db] = sorted(set(links))
    return out  # dict of db -> list of IDs

# 2) Get PMCID from PMID, if any
def pmid_to_pmcid(pmid: str):
    h = Entrez.elink(dbfrom="pubmed", id=pmid, linkname="pubmed_pmc")
    rec = Entrez.read(h)
    for lset in rec:
        for ldb in lset.get("LinkSetDb", []):
            if ldb.get("LinkName") == "pubmed_pmc":
                for link in ldb.get("Link", []):
                    return link["Id"]
    return None

# 3) From PMCID: list & download supplementary files hosted by PMC
def list_pmc_supp_files(pmcid: str):
    # Fetch JATS XML for the article
    h = Entrez.efetch(db="pmc", id=pmcid, rettype="full", retmode="xml")
    xml = h.read().decode("utf-8") if isinstance(h.read, bytes) else h.read()
    # Find supplementary-media hrefs (simple regex; for production use an XML parser)
    hrefs = set(re.findall(r'<(?:supplementary-material|media)[^>]*?xlink:href="([^"]+)"', xml))
    # Turn relative hrefs into absolute URLs
    base = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/"
    # Some files live under /articles/PMCID/bin/ too; we’ll try both
    supp_urls = []
    for href in hrefs:
        if href.startswith("http"):
            supp_urls.append(href)
        else:
            # common PMC paths
            supp_urls.append(f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/{href}")
            supp_urls.append(f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/bin/{href}")
    return sorted(set(supp_urls))

def download_files(urls, out_dir="supp_files"):
    out = []
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    for u in urls:
        fname = pathlib.Path(out_dir) / pathlib.Path(u.split("?")[0]).name
        try:
            r = requests.get(u, timeout=60)
            if r.ok and r.content:
                fname.write_bytes(r.content)
                out.append(str(fname))
        except Exception:
            pass
    return out

# 4) (Optional) Publisher full-text/supp links via llinks
def pubmed_provider_links(pmid: str):
    # returns provider landing pages; may include “supplementary” anchors depending on publisher
    h = Entrez.elink(dbfrom="pubmed", id=pmid, cmd="llinks")
    rec = Entrez.read(h)
    out = []
    print(rec)
    quit()
    for lset in rec:
        for prov in lset.get("IdUrlList", []):
            for item in prov.get("ObjUrl", []):
                url = item.get("Url")
                attrs = item.get("Attribute", [])
                out.append({"url": url, "attrs": attrs})
    return out

pmid = "40843950"  # example
pmcid = pmid_to_pmcid(pmid)
print(pmcid)

record = Entrez.read(Entrez.einfo())
print(record)