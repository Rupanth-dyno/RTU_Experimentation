
import json
from pathlib import Path


# Paths to the dataset files
claims_file = Path("claims_dev.jsonl")
corpus_file = Path("corpus.jsonl")


# Load claims
claims = []

with claims_file.open("r", encoding="utf-8") as file:
    for line in file:
        claims.append(json.loads(line))


# Load scientific documents
corpus = {}

with corpus_file.open("r", encoding="utf-8") as file:
    for line in file:
        document = json.loads(line)
        corpus[document["doc_id"]] = document


# Basic statistics
print(" DATASET STATISTICS ")

print(f"Total claims: {len(claims)}")
print(f"Total documents: {len(corpus)}")


claims_with_citations = 0
claims_with_evidence = 0
claims_with_available_documents = 0

for claim in claims:

    cited_doc_ids = claim.get("cited_doc_ids", [])

    if cited_doc_ids:
        claims_with_citations += 1

    if claim.get("evidence"):
        claims_with_evidence += 1

    available_documents = [
        doc_id for doc_id in cited_doc_ids
        if doc_id in corpus
    ]

    if available_documents:
        claims_with_available_documents += 1


print(f"Claims with citations: {claims_with_citations}")
print(f"Claims with evidence annotations: {claims_with_evidence}")
print(f"Claims with available source documents: {claims_with_available_documents}")


# Inspect claims containing evidence annotations
print("\n EXAMPLES WITH EVIDENCE ")

examples_found = 0

for claim in claims:

    if claim.get("evidence"):

        print("\nClaim ID:", claim["id"])
        print("Claim:", claim["claim"])
        print("Cited document IDs:", claim["cited_doc_ids"])
        print("Evidence annotations:")
        print(json.dumps(claim["evidence"], indent=2))

        examples_found += 1

        if examples_found == 3:
            break


# Inspect one scientific document
print("\n EXAMPLE DOCUMENT ")

first_document = next(iter(corpus.values()))

print("Document ID:", first_document["doc_id"])
print("Title:", first_document["title"])

print("\nAbstract sentences:")

for index, sentence in enumerate(first_document["abstract"]):
    print(f"{index}: {sentence}")
