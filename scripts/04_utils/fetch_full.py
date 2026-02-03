from Bio import Entrez
from lxml import etree
import os
import re

Entrez.email = "you@example.com"
if os.getenv("NCBI_API_KEY"):
    Entrez.api_key = os.getenv("NCBI_API_KEY")

def retrieve_pmc_fulltext(pmcid: str) -> dict:
    """
    Fetch and parse a PubMed Central (PMC) article in JATS XML format,
    extracting key sections into a JSON string.

    This function retrieves the full XML for the given PMC ID, parses it to
    extract the title, abstract, top-level body sections (with nested content
    flattened into text), and license URL. Tables, figures, formulas, and
    supplementary materials are excluded from the text extraction.

    Args:
        pmcid (str): The PMC identifier (e.g., "PMC11461603") for the article
            to fetch and parse.
    """
    
    if pmcid.startswith("PMC") == False:
        pmcid = "PMC" + pmcid

    # Fetch JATS XML from PMC
    h = Entrez.efetch(db="pmc", id=pmcid, rettype="full", retmode="xml")
    journal_xml_bytes = h.read()
    h.close()
    
    # Utils for parsing XML
    def _norm(s: str) -> str:
        return " ".join(s.split())

    def _xp(node, path, ns=None):
        return node.xpath(path, namespaces=ns or {})

    def _text_of(node):
        return _norm("".join(node.itertext()))

    def _title_text(node):
        t = _xp(node, "./*[local-name()='title']")
        return _norm("".join(t[0].itertext())) if t else ""

    def _get_attr_by_localname(el, localname: str):
        for k, v in el.attrib.items():
            if k.split('}')[-1] == localname:
                return v
        return None

    def _dedup_leading_title(title: str, content: str) -> str:
        if not title or not content:
            return content
        pattern = r"^\s*" + re.escape(title) + r"([.:;,-])?\s+"
        return re.sub(pattern, "", content, count=1, flags=re.IGNORECASE)

    def _strip_banned(node):
        _BANNED = {
            "table-wrap","table","thead","tbody","tr","td","th",
            "caption","fig","figure","graphic","media",
            "inline-graphic","disp-formula","inline-formula",
            "supplementary-material"
        }
        node_copy = etree.fromstring(etree.tostring(node))
        for bad in _xp(node_copy, ".//*"):
            if isinstance(bad.tag, str):
                ln = etree.QName(bad.tag).localname
                if ln in _BANNED:
                    parent = bad.getparent()
                    if parent is not None:
                        parent.remove(bad)
        return node_copy

    def _flatten_sec(sec_node) -> str:
        # Flatten section to text, excluding tables/figures
        parts = []
        this_title = _title_text(sec_node)

        for child in sec_node:
            if not isinstance(child.tag, str):
                continue
            ln = etree.QName(child.tag).localname
            if ln == "title":
                continue
            if ln == "sec":
                sub_txt = _flatten_sec(child)
                if sub_txt:
                    parts.append(sub_txt)
            else:
                stripped = _strip_banned(child)
                txt = _text_of(stripped).strip()
                if txt:
                    parts.append(txt)

        body = "\n".join(p for p in parts if p).strip()
        if this_title and body:
            body = _dedup_leading_title(this_title, body)
            return f"{this_title}\n{body}".strip()
        elif this_title:
            return this_title
        else:
            return body

    # Parse JATS XML to dict with title, abstract, sections, license
    parser = etree.XMLParser(recover=True)
    root = etree.fromstring(journal_xml_bytes, parser=parser)

    output_json = {}

    # Title
    title_nodes = _xp(root, ".//*[local-name()='article-title']")
    if title_nodes:
        output_json["Title"] = _text_of(title_nodes[0])

    # Abstract
    abs_nodes = _xp(root, ".//*[local-name()='abstract']")
    if abs_nodes:
        abstract_el = abs_nodes[0]
        secs = _xp(abstract_el, "./*[local-name()='sec']")
        if secs:
            parts = [_flatten_sec(sec) for sec in secs]
            output_json["Abstract"] = "\n".join(p for p in parts if p).strip()
        else:
            stripped = _strip_banned(abstract_el)
            t = _title_text(stripped)
            for tnode in _xp(stripped, ".//*[local-name()='title']"):
                parent = tnode.getparent()
                if parent is not None:
                    parent.remove(tnode)
            body = _text_of(stripped).strip()
            if t and body:
                body = _dedup_leading_title(t, body)
                output_json["Abstract"] = f"{t}\n{body}".strip()
            elif t:
                output_json["Abstract"] = t
            else:
                output_json["Abstract"] = body

    # Body sections
    first_level_secs = _xp(root, ".//*[local-name()='body']/*[local-name()='sec']")
    for sec in first_level_secs:
        key = _title_text(sec) or "UNTITLED"
        output_json[key] = _flatten_sec(sec)

    # License URL
    ns = {
        "xlink": "http://www.w3.org/1999/xlink",
        "ali": "http://www.niso.org/schemas/ali/1.0/",
    }
    lic_urls = []

    for n in _xp(root, ".//*[local-name()='permissions']//*[local-name()='license_ref'] | .//*[local-name()='license_ref']", ns):
        txt = _text_of(n)
        if txt.startswith(("http://", "https://")):
            lic_urls.append(txt)

    for n in _xp(root, ".//*[local-name()='permissions']//*[local-name()='license'] | .//*[local-name()='license']", ns):
        href = _get_attr_by_localname(n, "href")
        if href and href.startswith(("http://", "https://")):
            lic_urls.append(href)

    for n in _xp(root, ".//*[local-name()='permissions']//*[local-name()='license']//*[local-name()='ext-link']", ns):
        href = _get_attr_by_localname(n, "href")
        if href and href.startswith(("http://", "https://")):
            lic_urls.append(href)

    # Pick canonical license URL
    if lic_urls:
        seen, ordered = set(), []
        for u in lic_urls:
            if not u:
                continue
            u = u.strip()
            if u and u not in seen:
                seen.add(u)
                ordered.append(u)
        if ordered:
            for u in ordered:
                if "creativecommons.org/licenses" in u:
                    output_json["License"] = u
                    break
            else:
                output_json["License"] = ordered[0]
    return output_json



if __name__ == "__main__":
    pmcid = "PMC11461603"
    sections = retrieve_pmc_fulltext(pmcid)
    for k, v in sections.items():
        print(f"\n\n=== {k} ===\n{v}")