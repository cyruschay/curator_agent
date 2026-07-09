"""
Static alias dicts for high-frequency ontology entities.

Checked BEFORE exact-match dict and vector search so common lookups
bypass file I/O and embedding entirely.  Values are the full ontology
entry text — same format as the exact-match dicts in tools.py.
"""

# ---------------------------------------------------------------------------
# Disease Ontology (DOID)
# Keys: lowercased surface forms (including common synonyms)
# Values: full entry line from disease_ontology_v2.txt
# ---------------------------------------------------------------------------
DISEASE_ALIASES: dict[str, str] = {
    "cancer": "Cancer (DOID:162) is a disease of cellular proliferation that is malignant and primary, characterized by uncontrolled cellular proliferation, local cell invasion and metastasis. It has exact synonyms: malignant neoplasm; malignant tumor; primary cancer and related synonyms: No related synonyms. Cancer has parent term disease of cellular proliferation (DOID:14566).",
    "malignant neoplasm": "Cancer (DOID:162) is a disease of cellular proliferation that is malignant and primary, characterized by uncontrolled cellular proliferation, local cell invasion and metastasis. It has exact synonyms: malignant neoplasm; malignant tumor; primary cancer and related synonyms: No related synonyms. Cancer has parent term disease of cellular proliferation (DOID:14566).",
    "breast cancer": "Breast cancer (DOID:1612) (MIM:114480) is an organ system cancer that originates in the mammary gland. It has exact synonyms: breast tumor; malignant neoplasm of breast; malignant tumor of the breast; mammary cancer; mammary tumor; primary breast cancer and related synonyms: mammary neoplasm. Breast cancer has parent term organ system cancer (DOID:0050686).",
    "mammary cancer": "Breast cancer (DOID:1612) (MIM:114480) is an organ system cancer that originates in the mammary gland. It has exact synonyms: breast tumor; malignant neoplasm of breast; malignant tumor of the breast; mammary cancer; mammary tumor; primary breast cancer and related synonyms: mammary neoplasm. Breast cancer has parent term organ system cancer (DOID:0050686).",
    "lung cancer": "Lung cancer (DOID:1324) (MIM:211980) is a respiratory system cancer that is located_in the lung. It has exact synonyms: No exact synonyms and related synonyms: lung neoplasm. Lung cancer has parent term respiratory system cancer (DOID:0050615).",
    "colorectal cancer": "Colorectal cancer (DOID:9256) (MIM:114500) is a large intestine cancer that is located_in the colon and/or located_in the rectum. It has exact synonyms: No exact synonyms and related synonyms: No related synonyms. Colorectal cancer has parent term large intestine cancer (DOID:5672).",
    "colon cancer": "Colon cancer (DOID:219) is a colorectal cancer that is located_in the colon. It has exact synonyms: No exact synonyms and related synonyms: No related synonyms. Colon cancer has parent term colorectal cancer (DOID:9256).",
    "prostate cancer": "Prostate cancer (DOID:10283) (MIM:176807) is a male reproductive organ cancer that is located_in the prostate. It has exact synonyms: NGP - new growth of prostate; hereditary prostate cancer; malignant tumor of the prostate; prostate cancer, familial; prostate neoplasm; prostatic cancer; prostatic neoplasm; tumor of the prostate and related synonyms: No related synonyms. Prostate cancer has parent term male reproductive organ cancer (DOID:3856).",
    "hepatocellular carcinoma": "Hepatocellular carcinoma (DOID:684) (MIM:114550) is a liver carcinoma that has_material_basis_in undifferentiated hepatocytes and located_in the liver. It has exact synonyms: Hepatoma and related synonyms: No related synonyms. Hepatocellular carcinoma has parent term liver carcinoma (DOID:686).",
    "hepatoma": "Hepatocellular carcinoma (DOID:684) (MIM:114550) is a liver carcinoma that has_material_basis_in undifferentiated hepatocytes and located_in the liver. It has exact synonyms: Hepatoma and related synonyms: No related synonyms. Hepatocellular carcinoma has parent term liver carcinoma (DOID:686).",
    "hcc": "Hepatocellular carcinoma (DOID:684) (MIM:114550) is a liver carcinoma that has_material_basis_in undifferentiated hepatocytes and located_in the liver. It has exact synonyms: Hepatoma and related synonyms: No related synonyms. Hepatocellular carcinoma has parent term liver carcinoma (DOID:686).",
    "covid-19": "COVID-19 (DOID:0080600) is a Coronavirus infectious disease that is characterized by fever, cough and shortness of breath and that has_material_basis_in Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), a subtype of Betacoronavirus pandemicum. It has exact synonyms: 2019 Novel Coronavirus (2019-nCoV); 2019-nCoV infection; COVID19; SARS-CoV-2 infection; Wuhan coronavirus infection; Wuhan seafood market pneumonia virus infection and related synonyms: No related synonyms. COVID-19 has parent term Coronavirus infectious disease (DOID:0080599).",
    "sars-cov-2 infection": "COVID-19 (DOID:0080600) is a Coronavirus infectious disease that is characterized by fever, cough and shortness of breath and that has_material_basis_in Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), a subtype of Betacoronavirus pandemicum. It has exact synonyms: 2019 Novel Coronavirus (2019-nCoV); 2019-nCoV infection; COVID19; SARS-CoV-2 infection; Wuhan coronavirus infection; Wuhan seafood market pneumonia virus infection and related synonyms: No related synonyms. COVID-19 has parent term Coronavirus infectious disease (DOID:0080599).",
    "liver cirrhosis": "Liver cirrhosis (DOID:5082) is  It has exact synonyms: Cirrhosis; cirrhosis of liver and related synonyms: No related synonyms. Liver cirrhosis has parent term liver disease (DOID:409).",
    "cirrhosis": "Liver cirrhosis (DOID:5082) is  It has exact synonyms: Cirrhosis; cirrhosis of liver and related synonyms: No related synonyms. Liver cirrhosis has parent term liver disease (DOID:409).",
    "type 2 diabetes mellitus": "Type 2 diabetes mellitus (DOID:9352) (MIM:125853) is a diabetes mellitus that is characterized by high blood sugar, insulin resistance, and relative lack of insulin. It has exact synonyms: NIDDM; insulin resistance; non-insulin-dependent diabetes mellitus; type 2 diabetes; type II diabetes mellitus and related synonyms: No related synonyms. Type 2 diabetes mellitus has parent term diabetes mellitus (DOID:9351).",
    "type 2 diabetes": "Type 2 diabetes mellitus (DOID:9352) (MIM:125853) is a diabetes mellitus that is characterized by high blood sugar, insulin resistance, and relative lack of insulin. It has exact synonyms: NIDDM; insulin resistance; non-insulin-dependent diabetes mellitus; type 2 diabetes; type II diabetes mellitus and related synonyms: No related synonyms. Type 2 diabetes mellitus has parent term diabetes mellitus (DOID:9351).",
    "pancreatic cancer": "Pancreatic cancer (DOID:1793) is an endocrine gland cancer located_in the pancreas. It has exact synonyms: Ca body of pancreas; Ca head of pancreas; Ca tail of pancreas; malignant neoplasm of body of pancreas; malignant neoplasm of head of pancreas; malignant neoplasm of tail of pancreas; pancreas neoplasm; pancreatic neoplasm; pancreatic tumor and related synonyms: No related synonyms. Pancreatic cancer has parent term endocrine gland cancer (DOID:170).",
    "systemic lupus erythematosus": "Systemic lupus erythematosus (DOID:9074) (MIM:152700) is a lupus erythematosus that is an inflammation of connective tissue marked by skin rashes, joint pain and swelling, inflammation of the kidneys and inflammation of the tissue surrounding the heart. It has exact synonyms: Lupus Erythematosus, systemic; SLE - Lupus Erythematosus, systemic; disseminated lupus erythematosus and related synonyms: No related synonyms. Systemic lupus erythematosus has parent term lupus erythematosus (DOID:8857).",
    "sle": "Systemic lupus erythematosus (DOID:9074) (MIM:152700) is a lupus erythematosus that is an inflammation of connective tissue marked by skin rashes, joint pain and swelling, inflammation of the kidneys and inflammation of the tissue surrounding the heart. It has exact synonyms: Lupus Erythematosus, systemic; SLE - Lupus Erythematosus, systemic; disseminated lupus erythematosus and related synonyms: No related synonyms. Systemic lupus erythematosus has parent term lupus erythematosus (DOID:8857).",
    "ovarian cancer": "Ovarian cancer (DOID:2394) (MIM:167000) is a female reproductive organ cancer that is located_in the ovary. It has exact synonyms: malignant Ovarian tumor; malignant tumour of ovary; ovarian neoplasm; ovary neoplasm; primary ovarian cancer; tumor of the Ovary and related synonyms: No related synonyms. Ovarian cancer has parent term female reproductive organ cancer (DOID:120).",
    "gastric cancer": "Stomach cancer (DOID:10534) (MIM:613659) is a gastrointestinal system cancer that is located_in the stomach. It has exact synonyms: gastric cancer; gastric neoplasm and related synonyms: No related synonyms. Stomach cancer has parent term gastrointestinal system cancer (DOID:3119).",
    "stomach cancer": "Stomach cancer (DOID:10534) (MIM:613659) is a gastrointestinal system cancer that is located_in the stomach. It has exact synonyms: gastric cancer; gastric neoplasm and related synonyms: No related synonyms. Stomach cancer has parent term gastrointestinal system cancer (DOID:3119).",
    "alzheimer's disease": "Alzheimer's disease (DOID:10652) is a tauopathy that is characterized by memory lapses, confusion, emotional instability and progressive loss of mental ability and results in progressive memory loss, impaired thinking, disorientation, and changes in personality and mood starting and leads in advanced cases to a profound decline in cognitive and physical functioning and is marked histologically by the degeneration of brain neurons especially in the cerebral cortex and by the presence of neurofibrillary tangles and plaques containing beta-amyloid. It has exact synonyms: Alzheimer disease; Alzheimers dementia and related synonyms: No related synonyms. Alzheimer's disease has parent term tauopathy (DOID:680).",
    "alzheimer disease": "Alzheimer's disease (DOID:10652) is a tauopathy that is characterized by memory lapses, confusion, emotional instability and progressive loss of mental ability and results in progressive memory loss, impaired thinking, disorientation, and changes in personality and mood starting and leads in advanced cases to a profound decline in cognitive and physical functioning and is marked histologically by the degeneration of brain neurons especially in the cerebral cortex and by the presence of neurofibrillary tangles and plaques containing beta-amyloid. It has exact synonyms: Alzheimer disease; Alzheimers dementia and related synonyms: No related synonyms. Alzheimer's disease has parent term tauopathy (DOID:680).",
    "glioblastoma": "Glioblastoma (DOID:3068) is a malignant astrocytoma characterized by the presence of small areas of necrotizing tissue that is surrounded by anaplastic cells as well as the presence of hyperplastic blood vessels, and that has_material_basis_in abnormally proliferating cells derives_from multiple cell types including astrocytes and oligondroctyes. It has exact synonyms: GBM; adult glioblastoma multiforme; glioblastoma multiforme; grade IV adult Astrocytic tumor; primary glioblastoma multiforme; spongioblastoma multiforme and related synonyms: No related synonyms. Glioblastoma has parent term malignant astrocytoma (DOID:3069).",
    "gbm": "Glioblastoma (DOID:3068) is a malignant astrocytoma characterized by the presence of small areas of necrotizing tissue that is surrounded by anaplastic cells as well as the presence of hyperplastic blood vessels, and that has_material_basis_in abnormally proliferating cells derives_from multiple cell types including astrocytes and oligondroctyes. It has exact synonyms: GBM; adult glioblastoma multiforme; glioblastoma multiforme; grade IV adult Astrocytic tumor; primary glioblastoma multiforme; spongioblastoma multiforme and related synonyms: No related synonyms. Glioblastoma has parent term malignant astrocytoma (DOID:3069).",
    "atrial fibrillation": "Atrial fibrillation (DOID:0060224) is a heart conduction disease that is characterized by uncoordinated electrical activity in the heart's upper chambers (the atria), which causes the heartbeat to become fast and irregular and has symptoms palpitations, weakness, fatigue, lightheadedness, dizziness, confusion, shortness of breath and chest pain. It has exact synonyms: A-fib; AFib and related synonyms: No related synonyms. Atrial fibrillation has parent term heart conduction disease (DOID:10273).",
    "osteoarthritis": "Osteoarthritis (DOID:8398) is an arthritis that has_material_basis_in worn out cartilage located_in joint. It has exact synonyms: Osteoarthrosis and allied disorder; degenerative arthritis; degenerative joint disease; hypertrophic arthritis; osteoarthrosis and related synonyms: No related synonyms. Osteoarthritis has parent term arthritis (DOID:848).",
}

# ---------------------------------------------------------------------------
# Species / Taxonomy (NCBI TaxID)
# Keys: lowercased surface forms
# Values: pre-formatted TSV row  (tax_id \t scientific_name \t common_name \t includes \t genbank_common_name)
# ---------------------------------------------------------------------------
SPECIES_ALIASES: dict[str, str] = {
    "human": "9606\tHomo sapiens\t(None)\t(None)\thuman",
    "humans": "9606\tHomo sapiens\t(None)\t(None)\thuman",
    "homo sapiens": "9606\tHomo sapiens\t(None)\t(None)\thuman",
    "patient": "9606\tHomo sapiens\t(None)\t(None)\thuman",
    "patients": "9606\tHomo sapiens\t(None)\t(None)\thuman",
    "mouse": "10090\tMus musculus\tmouse\tBalb/c mouse, LK3 transgenic mice, Mus sp. 129SV, nude mice, transgenic mice\thouse mouse",
    "mice": "10090\tMus musculus\tmouse\tBalb/c mouse, LK3 transgenic mice, Mus sp. 129SV, nude mice, transgenic mice\thouse mouse",
    "mus musculus": "10090\tMus musculus\tmouse\tBalb/c mouse, LK3 transgenic mice, Mus sp. 129SV, nude mice, transgenic mice\thouse mouse",
    "rat": "10116\tRattus norvegicus\t(None)\t(None)\tNorway rat",
    "rats": "10116\tRattus norvegicus\t(None)\t(None)\tNorway rat",
    "rattus norvegicus": "10116\tRattus norvegicus\t(None)\t(None)\tNorway rat",
    "horse": "9796\tEquus caballus\t(None)\t(None)\thorse",
    "equus caballus": "9796\tEquus caballus\t(None)\t(None)\thorse",
    "pig": "9823\tSus scrofa\tpig\t(None)\twild boar",
    "sus scrofa": "9823\tSus scrofa\tpig\t(None)\twild boar",
    "chicken": "9031\tGallus gallus\t(None)\t(None)\tchicken",
    "gallus gallus": "9031\tGallus gallus\t(None)\t(None)\tchicken",
    "rabbit": "9986\tOryctolagus cuniculus\t(None)\t(None)\trabbit",
    "dog": "9615\tCanis lupus familiaris\t(None)\t(None)\tdog",
    "bovine": "9913\tBos taurus\t(None)\t(None)\tcattle",
    "cow": "9913\tBos taurus\t(None)\t(None)\tcattle",
    "bos taurus": "9913\tBos taurus\t(None)\t(None)\tcattle",
    "zebrafish": "7955\tDanio rerio\t(None)\t(None)\tzebrafish",
    "danio rerio": "7955\tDanio rerio\t(None)\t(None)\tzebrafish",
    "drosophila": "7227\tDrosophila melanogaster\t(None)\t(None)\tfruit fly",
    "drosophila melanogaster": "7227\tDrosophila melanogaster\t(None)\t(None)\tfruit fly",
    "c. elegans": "6239\tCaenorhabditis elegans\t(None)\t(None)\tnematode",
    "caenorhabditis elegans": "6239\tCaenorhabditis elegans\t(None)\t(None)\tnematode",
    "e. coli": "562\tEscherichia coli\t(None)\t(None)\tE. coli",
    "escherichia coli": "562\tEscherichia coli\t(None)\t(None)\tE. coli",
    "rhesus macaque": "9544\tMacaca mulatta\t(None)\t(None)\trhesus macaque",
    "macaca mulatta": "9544\tMacaca mulatta\t(None)\t(None)\trhesus macaque",
}

# ---------------------------------------------------------------------------
# Specimen / Anatomy (UBERON / CL)
# Keys: lowercased surface forms
# Values: full entry line from uberon_terms.txt
# ---------------------------------------------------------------------------
SPECIMEN_ALIASES: dict[str, str] = {
    "serum": "Blood serum (UBERON:0001977) - The portion of blood plasma that excludes clotting factors. Synonyms: serum. It has parent term(s): haemolymphatic fluid (UBERON:0000179).",
    "blood serum": "Blood serum (UBERON:0001977) - The portion of blood plasma that excludes clotting factors. Synonyms: serum. It has parent term(s): haemolymphatic fluid (UBERON:0000179).",
    "sera": "Blood serum (UBERON:0001977) - The portion of blood plasma that excludes clotting factors. Synonyms: serum. It has parent term(s): haemolymphatic fluid (UBERON:0000179).",
    "plasma": "Blood plasma (UBERON:0001969) - The liquid component of blood, in which erythrocytes are suspended. Synonyms: portion of plasma; plasma. It has parent term(s): haemolymphatic fluid (UBERON:0000179); organism substance (UBERON:0000463).",
    "blood plasma": "Blood plasma (UBERON:0001969) - The liquid component of blood, in which erythrocytes are suspended. Synonyms: portion of plasma; plasma. It has parent term(s): haemolymphatic fluid (UBERON:0000179); organism substance (UBERON:0000463).",
    "blood": "Blood (UBERON:0000178) - A fluid that is composed of blood plasma and erythrocytes. Synonyms: whole blood. It has parent term(s): haemolymphatic fluid (UBERON:0000179).",
    "whole blood": "Blood (UBERON:0000178) - A fluid that is composed of blood plasma and erythrocytes. Synonyms: whole blood. It has parent term(s): haemolymphatic fluid (UBERON:0000179).",
    "urine": "Urine (UBERON:0001088) - Excretion that is the output of a kidney. Synonyms: none. It has parent term(s): excreta (UBERON:0000174).",
    "saliva": "Saliva (UBERON:0001836) - A fluid produced in the oral cavity by salivary glands, typically used in predigestion, but also in other functions. Synonyms: saliva atomaris; saliva molecularis; sailva normalis. It has parent term(s): secretion of exocrine gland (UBERON:0000456).",
    "csf": "Cerebrospinal fluid (UBERON:0001359) - A clear, colorless, bodily fluid, that occupies the subarachnoid space and the ventricular system around and inside the brain and spinal cord. Synonyms: liquor cerebrospinalis; spinal fluid. It has parent term(s): transudate (UBERON:0007779).",
    "cerebrospinal fluid": "Cerebrospinal fluid (UBERON:0001359) - A clear, colorless, bodily fluid, that occupies the subarachnoid space and the ventricular system around and inside the brain and spinal cord. Synonyms: liquor cerebrospinalis; spinal fluid. It has parent term(s): transudate (UBERON:0007779).",
    "tissue": "Tissue (UBERON:0000479) - Multicellular anatomical structure that consists of many cells of one or a few types, arranged in an extracellular matrix such that their long-range organisation is at least partly a repetition of their short-range organisation. Synonyms: simple tissue. It has parent term(s): multicellular anatomical structure (UBERON:0010000).",
    "liver": "Liver (UBERON:0002107) - An exocrine gland which secretes bile and functions in metabolism of protein and carbohydrate and fat, synthesizes substances involved in the clotting of the blood, synthesizes vitamin A, detoxifies poisonous substances, stores glycogen, and breaks down worn-out erythrocytes[GO]. Synonyms: iecur; jecur. It has parent term(s): digestive system gland (UBERON:0006925); endoderm-derived structure (UBERON:0004119); exocrine gland (UBERON:0002365); abdomen element (UBERON:0005172).",
    "kidney": "Kidney (UBERON:0002113) - A paired organ of the urinary tract that produces urine and maintains bodily fluid homeostasis, blood pressure, pH levels, red blood cell production and skeleton mineralization. Synonyms: reniculate kidney. It has parent term(s): lateral structure (UBERON:0015212); cavitated compound organ (UBERON:0000489); abdomen element (UBERON:0005172).",
    "brain": "Brain (UBERON:0000955) - The brain is the center of the nervous system in all vertebrate, and most invertebrate, animals. Synonyms: suprasegmental levels of nervous system; suprasegmental structures; synganglion; encephalon; the brain. It has parent term(s): organ (UBERON:0000062); ectoderm-derived structure (UBERON:0004121).",
    "colon": "Colon (UBERON:0001155) - A portion of the large intestine before it becomes the rectum. Synonyms: hindgut. It has parent term(s): subdivision of digestive tract (UBERON:0004921).",
    "heart": "Heart (UBERON:0000948) - A myogenic muscular circulatory organ found in the vertebrate cardiovascular system composed of chambers of cardiac muscle. Synonyms: cardium. It has parent term(s): mesoderm-derived structure (UBERON:0004120); thoracic segment organ (UBERON:0005181); primary circulatory organ (UBERON:0007100); structure with developmental contribution from neural crest (UBERON:0010314).",
    "lung": "Lung (UBERON:0002048) - Respiration organ that develops as an outpocketing of the esophagus. Synonyms: pulmo. It has parent term(s): lateral structure (UBERON:0015212); respiration organ (UBERON:0000171); thoracic cavity element (UBERON:0005178).",
    "pancreas": "Pancreas (UBERON:0001264) - An endoderm derived structure that produces precursors of digestive enzymes and blood glucose regulating hormones[GO]. Synonyms: none. It has parent term(s): viscus (UBERON:0002075).",
    "stomach": "Stomach (UBERON:0000945) - An expanded region of the vertebrate alimentary tract that serves as a food storage compartment and digestive organ. Synonyms: anterior intestine; ventriculus; gaster; mesenteron. It has parent term(s): subdivision of digestive tract (UBERON:0004921).",
    "synovial fluid": "Synovial fluid (UBERON:0001090) - Joint fluid is a transudate of plasma that is actively secreted by synovial cells. Synonyms: joint fluid. It has parent term(s): secretion of serous gland (UBERON:0007794); transudate (UBERON:0007779).",
    "breast": "Breast (UBERON:0000310) - The upper ventral region of the torso of an organism. Synonyms: mamma. It has parent term(s): external soft tissue zone (UBERON:0034929).",
}
