from typing import List

def onto_taxonomy_tool(organism_names: List[str]) -> str:
    from Bio import Entrez
    
    # Set email for NCBI Entrez (required by NCBI)
    if not Entrez.email:
        Entrez.email = "glycan.curator@example.com"
    
    if not organism_names:
        return "Error: No organism names provided"
    
    output_str = ""
    
    for name in organism_names:
        query_name = name.strip()
        output_str += f"{query_name}:\ntax_id\tscientific_name\tcommon_name\tincludes\tgenbank_common_name\n"
        
        ### Common organisms
        name_human = ["human", "humans", "homo sapiens"]
        name_mouse = ["mouse", "mice", "mus musculus"]
        name_rat = ["rat", "rats", "rattus norvegicus"]
        if query_name.lower in name_human:
            output_str += "9606\tHomo sapiens\t(None)\t(None)\thuman"
            continue
        if query_name.lower in name_mouse:
            output_str += "10090\tMus musculus\tmouse\tBalb/c mouse, LK3 transgenic mice, Mus sp. 129SV, nude mice, transgenic mice\thouse mouse"
            continue
        if query_name.lower in name_rat:
            output_str += "10116\tRattus norvegicus\tbrown rat, rat, rats\tBuffalo rat, Rattus PC12 clone IS, Rattus sp. strain Wistar, Sprague-Dawley rat, Wistar rats, laboratory rat, zitter rats\tNorway rat"
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

if __name__ == "__main__":
    # Example usage
    test_organisms = ["rat"]
    result = onto_taxonomy_tool(test_organisms)
    print(result)
