INPUT_SCHEMA = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "processing_id": {
      "type": "string"
    },
    "pmid": {
      "type": "string"
    },
    "pmcid": {
      "type": ["string", "null"]
    },
    "title": {
      "type": "string"
    },
    "abstract": {
      "type": ["string", "null"]
    },
    "keywords": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "chemical_names": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "chemical_rn": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "chemical_uid": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "mesh_names": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "mesh_uid": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "journal_abbr": {
      "type": "string"
    },
    "article_date": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "Year": {
            "type": "string"
          },
          "Month": {
            "type": "string"
          },
          "Day": {
            "type": "string"
          }
        },
        "required": ["Year", "Month", "Day"]
      }
    }
  },
  "required": ["processing_id", "pmid", "title", "abstract"]
}