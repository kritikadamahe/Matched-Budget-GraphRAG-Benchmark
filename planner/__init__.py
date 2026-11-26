"""
GraphRAG budget planner: "how much of my corpus is worth extracting?"

    ingest     uploaded files (txt, md, pdf, html, csv, json) -> documents
    costs      extraction / query cost model, calibrated on real extraction calls
    questions  LLM-generated test questions with known source chunks (gold evidence)
    pilot      run the matched-budget pipeline on a sample, project to the full corpus, recommend
    sample     the same report built from the logged 75-question HotpotQA benchmark (no API calls)

Used by demo/server.py (the "Budget planner" tab).
"""
