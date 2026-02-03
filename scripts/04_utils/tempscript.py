from Bio import Entrez
from lxml import etree
import os
import re
import json

Entrez.email = "you@example.com"
if os.getenv("NCBI_API_KEY"):
    Entrez.api_key = os.getenv("NCBI_API_KEY")

def retrieve_pmc_fulltext(pmcid: str) -> dict:
    """
    Fetch and parse a PubMed Central (PMC) article in JATS XML format,
    extracting key sections into a nested dictionary.

    This function retrieves the full XML for the given PMC ID, parses it to
    extract the title, abstract, top-level body sections (with nested subsections
    and paragraphs), and license URL. Tables, figures, formulas, and
    supplementary materials are excluded from the text extraction. The structure
    is nested: sections and subsections are dictionaries with 'title' and 'content'
    keys, where 'content' is a list of strings (paragraphs) or further subsection dicts.

    Args:
        pmcid (str): The PMC identifier (e.g., "PMC11461603") for the article
            to fetch and parse.
    """
    
    if not pmcid.startswith("PMC"):
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

    def is_subtitle(s: str) -> bool:
        s = s.strip()
        return ('.' not in s and '!' not in s and '?' not in s and len(s) > 5)

    def nest_content(content_list: list) -> list:
        new_content = []
        current_sub = None
        for item in content_list:
            if isinstance(item, dict):
                if current_sub:
                    new_content.append(current_sub)
                new_content.append(item)
                current_sub = None
                continue
            # str
            if is_subtitle(item):
                if current_sub:
                    new_content.append(current_sub)
                current_sub = {"title": item, "content": []}
            else:
                if current_sub:
                    current_sub["content"].append(item)
                else:
                    new_content.append(item)
        if current_sub:
            new_content.append(current_sub)
        return new_content

    def parse_sec(node) -> dict:
        title = _title_text(node)
        content = []
        for child in node:
            if not isinstance(child.tag, str):
                continue
            ln = etree.QName(child.tag).localname
            if ln == "title":
                continue
            if ln == "sec":
                sub = parse_sec(child)
                content.append(sub)
            else:
                stripped = _strip_banned(child)
                txt = _text_of(stripped).strip()
                if txt:
                    content.append(txt)
        # Dedup leading title from first paragraph if applicable
        if content and isinstance(content[0], str) and title:
            content[0] = _dedup_leading_title(title, content[0])
        # Nest detected subsections
        content = nest_content(content)
        return {"title": title, "content": content}

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
        output_json["Abstract"] = parse_sec(abs_nodes[0])

    # Body sections
    first_level_secs = _xp(root, ".//*[local-name()='body']/*[local-name()='sec']")
    for sec in first_level_secs:
        sec_dict = parse_sec(sec)
        key = sec_dict["title"] or "UNTITLED"
        output_json[key] = sec_dict

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
    pmcid = "PMC12379856"
    sections = retrieve_pmc_fulltext(pmcid)
    print(json.dumps(sections.get("License"), indent=2, ensure_ascii=False))