# RAG evaluation status

## Implementation found

`ner_rag_pipeline.py` contains eight hard-coded short passages, TF-IDF cosine retrieval, and regex extraction. Retrieval returns the local `doc_id`, title, text, and score, so the passage can be inspected. The passages have no URL, DOI, journal, publication year, page, or section. The file does not establish that these passages or their numerical values came from identifiable publications. They are therefore an unverified demonstration corpus, not scientific evidence.

The reported 87.5% extraction accuracy compares regex output with annotations attached to the same eight passages (four fields each). It is a small self-annotated extraction check; it does not measure retrieval recall, independent factual correctness, citation correctness, answer support, or abstention in conflicting-source cases. No external LLM is used.

## Safe evaluation boundary

Do not use the current passage values as citations or merge them with JARVIS records. A scientific evaluation set is deferred until source documents can be identified and checked. For each future example, record:

| Field | Required content |
| --- | --- |
| question | User-style question and scope |
| expected answer | Manually checked answer, units and value type (DFT, experiment, or model prediction) |
| evidence | Exact supporting passage and location |
| source | Title plus verified URL/DOI and document metadata |
| expected citations | Source identifiers that must be returned |
| case type | Direct lookup, comparison, prediction, missing evidence, conflict/units |
| abstention rule | What evidence is insufficient or contradictory |

Evaluate retrieval relevance/recall separately from answer correctness and citation support. Keep synthetic test passages isolated and visibly labelled. Current scientific RAG quality is **not evaluated**.
