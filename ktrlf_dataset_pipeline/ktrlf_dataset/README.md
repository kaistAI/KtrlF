# Ktrl+F Dataset
The Ktrlf dataset includes QA pairs for each document in every line. 

- `id`: URL of the document extracted from C4
- `target_text`: The document used for In-Document Search
- `qa_pairs`: Pairs of questions and their corresponding answer entities
- `entity_info`: Metadata extracted from the document using GCP
  - `mention`: The text mentioned in the actual document
  - `entity`: The linked wiki entity
  - `start`: The start character indices in the document
  - `end`: The end character indices in the document

### Data Example
```
{
    'id': <Dump ULR of Document>,
    'data': {
        'qa_pairs': [
            {'question': 'Social media platforms', 'target_entities': ['Twitter']},
            {'question': '...', 'target_entities': ['...']},
            ...
        ],
        'target_text': '...',
        'entity_info': [
            {'mention': 'Trump',
                'entity': 'Donald Trump',
                'start': 11,
                'end': 16,
                'wikipedia_link': 'https://en.wikipedia.org/wiki/Donald_Trump',
                'gcp_entity_type': 'Type.ORGANIZATION'},
            {'mention': 'Democratic',
                'entity': 'Democratic Party (United States)',
                'start': 179,
                'end': 189,
                'wikipedia_link': 'https://en.wikipedia.org/wiki/Democratic_Party_(United_States)',
                'gcp_entity_type': 'Type.ORGANIZATION'},
            ...
        ]
    }
}
```
