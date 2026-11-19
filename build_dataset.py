
import json
from pathlib import Path



claims_file = Path("claims_dev.jsonl")
corpus_file = Path("corpus.jsonl")

claims = []

with claims_file.open("r", encoding="utf-8") as file:
    for line in file:
        claims.append(json.loads(line))


corpus = {}

with corpus_file.open("r", encoding="utf-8") as file:
    for line in file:
        document = json.loads(line)

        corpus[document["doc_id"]] = document



processed_examples = []




for claim_data in claims:

    claim_id = claim_data["id"]
    claim_text = claim_data["claim"]

    cited_doc_ids = claim_data.get("cited_doc_ids", [])
    evidence_annotations = claim_data.get("evidence", {})


    # Only process claims with evidence annotations
    if not evidence_annotations:
        continue




    for cited_doc_id in cited_doc_ids:

        # Find corresponding document
        if cited_doc_id not in corpus:
            continue


        document = corpus[cited_doc_id]

        # Check whether evidence exists for this document
        document_evidence = evidence_annotations.get(
            str(cited_doc_id),
            evidence_annotations.get(cited_doc_id, [])
        )

        if not document_evidence:
            continue



        abstract_sentences = document["abstract"]

        candidate_passages = []

        for sentence_index, sentence in enumerate(abstract_sentences):

            candidate_passages.append({
                "passage_id": sentence_index,
                "text": sentence
            })



        evidence_groups = []

        gold_evidence_passage_ids = set()

        for evidence_group in document_evidence:

            sentence_indices = evidence_group["sentences"]

            label = evidence_group["label"]

            evidence_groups.append({
                "sentence_indices": sentence_indices,
                "label": label
            })

            for sentence_index in sentence_indices:

                gold_evidence_passage_ids.add(sentence_index)



        evidence_labels = sorted(
            set(group["label"] for group in document_evidence)
        )


        

        example = {

            "claim_id": claim_id,

            "claim": claim_text,

            "source_document_id": cited_doc_id,

            "source_title": document["title"],

            "source_abstract": abstract_sentences,

            "candidate_passages": candidate_passages,

            "evidence_groups": evidence_groups,

            "gold_evidence_passage_ids": sorted(
                gold_evidence_passage_ids
            ),

            "evidence_labels": evidence_labels

        }


        processed_examples.append(example)



output_directory = Path("processed")

output_directory.mkdir(exist_ok=True)


output_file = output_directory / "ranking_dataset.jsonl"


with output_file.open("w", encoding="utf-8") as file:

    for example in processed_examples:

        file.write(
            json.dumps(example, ensure_ascii=False) + "\n"
        )



print(" PROCESSED DATASET ")

print("Original claims:", len(claims))

print("Processed claim-document examples:",
      len(processed_examples))

print("Output file:", output_file)


print("\n EXAMPLE ")

if processed_examples:

    print(
        json.dumps(
            processed_examples[0],
            indent=2,
            ensure_ascii=False
        )
    )
