
import json


# Load claims
claims = []

with open("claims_dev.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        claims.append(json.loads(line))


# Load corpus
corpus = {}

with open("corpus.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        document = json.loads(line)
        corpus[document["doc_id"]] = document


# Statistics
label_counts = {}
claims_with_evidence = 0
total_evidence_groups = 0
total_evidence_sentences = 0
multiple_evidence_groups = 0
invalid_sentence_indices = 0
evidence_not_in_citations = 0


for claim in claims:

    evidence = claim.get("evidence", {})

    if not evidence:
        continue

    claims_with_evidence += 1

    cited_doc_ids = claim.get("cited_doc_ids", [])

    for doc_id, evidence_groups in evidence.items():

        # Count evidence labels
        for group in evidence_groups:

            label = group["label"]

            label_counts[label] = (
                label_counts.get(label, 0) + 1
            )

            total_evidence_groups += 1

            sentence_indices = group["sentences"]

            total_evidence_sentences += len(sentence_indices)

            if len(sentence_indices) > 1:
                multiple_evidence_groups += 1

            # Check whether evidence document is cited
            if int(doc_id) not in cited_doc_ids:
                evidence_not_in_citations += 1

            # Check sentence indices
            if int(doc_id) in corpus:

                abstract = corpus[int(doc_id)]["abstract"]

                for sentence_index in sentence_indices:

                    if sentence_index < 0 or sentence_index >= len(abstract):
                        invalid_sentence_indices += 1


print("ANNOTATION ANALYSIS")

print("Claims with evidence:", claims_with_evidence)

print("\nEvidence labels:")

for label, count in label_counts.items():
    print(f"{label}: {count}")

print("\nTotal evidence groups:", total_evidence_groups)

print("Total annotated evidence sentences:", total_evidence_sentences)

print("Evidence groups containing multiple sentences:", multiple_evidence_groups)

print("Evidence documents not included in cited_doc_ids:", evidence_not_in_citations)

print("Invalid sentence indices:", invalid_sentence_indices)
