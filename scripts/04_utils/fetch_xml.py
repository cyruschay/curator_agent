from Bio import Entrez

def fetch_xml(pmids):
    handle = Entrez.efetch(db="pubmed", id=",".join(pmids), retmode="xml")
    xml = Entrez.read(handle)
    handle.close()
    return xml

print(fetch_xml(["40792479"]))