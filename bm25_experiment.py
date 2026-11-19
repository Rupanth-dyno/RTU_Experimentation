
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi




dataset_file = Path("processed/ranking_dataset.jsonl")

examples = []

with dataset_file.open("r", encoding="utf-8") as file:

    for line in file:
        examples.append(json.loads(line))




def tokenize(text):

    text = text.lower()

    tokens = re.findall(r"\b\w+\b", text)

    return tokens




def recall_at_k(ranked_passage_ids, gold_passage_ids, k):

    retrieved = set(ranked_passage_ids[:k])

    gold = set(gold_passage_ids)

    if not gold:
        return 0.0

    return len(retrieved.intersection(gold)) / len(gold)


def reciprocal_rank(ranked_passage_ids, gold_passage_ids):

    gold = set(gold_passage_ids)

    for rank, passage_id in enumerate(
        ranked_passage_ids,
        start=1
    ):

        if passage_id in gold:
            return 1.0 / rank

    return 0.0




results = []

for example in examples:

    claim = example["claim"]

    passages = example["candidate_passages"]

    gold_passage_ids = example[
        "gold_evidence_passage_ids"
    ]


    # Tokenize candidate passages
    tokenized_passages = [
        tokenize(passage["text"])
        for passage in passages
    ]


    # Create BM25 model for this source document
    bm25 = BM25Okapi(tokenized_passages)


    # Tokenize claim
    tokenized_claim = tokenize(claim)


    # Calculate BM25 scores
    scores = bm25.get_scores(tokenized_claim)


    # Sort passages by score, highest first
    ranked_indices = sorted(
        range(len(passages)),
        key=lambda index: scores[index],
        reverse=True
    )


    # Convert ranked indices to passage IDs
    ranked_passage_ids = [
        passages[index]["passage_id"]
        for index in ranked_indices
    ]


    # Store rankings and scores
    ranking = []

    for rank, index in enumerate(ranked_indices, start=1):

        ranking.append({

            "rank": rank,

            "passage_id": passages[index]["passage_id"],

            "score": float(scores[index]),

            "text": passages[index]["text"],

            "is_gold_evidence":
                passages[index]["passage_id"]
                in gold_passage_ids

        })


    # Calculate metrics
    metrics = {

        "recall_at_1": recall_at_k(
            ranked_passage_ids,
            gold_passage_ids,
            1
        ),

        "recall_at_3": recall_at_k(
            ranked_passage_ids,
            gold_passage_ids,
            3
        ),

        "recall_at_5": recall_at_k(
            ranked_passage_ids,
            gold_passage_ids,
            5
        ),

        "mrr": reciprocal_rank(
            ranked_passage_ids,
            gold_passage_ids
        )

    }


    results.append({

        "claim_id": example["claim_id"],

        "source_document_id":
            example["source_document_id"],

        "claim": claim,

        "gold_evidence_passage_ids":
            gold_passage_ids,

        "ranking": ranking,

        "metrics": metrics

    })




number_of_examples = len(results)


average_recall_at_1 = sum(
    result["metrics"]["recall_at_1"]
    for result in results
) / number_of_examples


average_recall_at_3 = sum(
    result["metrics"]["recall_at_3"]
    for result in results
) / number_of_examples


average_recall_at_5 = sum(
    result["metrics"]["recall_at_5"]
    for result in results
) / number_of_examples


average_mrr = sum(
    result["metrics"]["mrr"]
    for result in results
) / number_of_examples



print(" BM25 RESULTS ")

print("Number of examples:", number_of_examples)

print(
    "Recall@1:",
    round(average_recall_at_1, 4)
)

print(
    "Recall@3:",
    round(average_recall_at_3, 4)
)

print(
    "Recall@5:",
    round(average_recall_at_5, 4)
)

print(
    "MRR:",
    round(average_mrr, 4)
)



output_directory = Path("results")

output_directory.mkdir(exist_ok=True)


output_file = output_directory / "bm25_results.jsonl"


with output_file.open("w", encoding="utf-8") as file:

    for result in results:

        file.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )


print("\nDetailed results saved to:", output_file)



example_result = results[0]


print("\n EXAMPLE RANKING ")

print("Claim:", example_result["claim"])

print(
    "Gold evidence:",
    example_result["gold_evidence_passage_ids"]
)


for passage in example_result["ranking"][:5]:

    print("\nRank:", passage["rank"])

    print("Passage ID:", passage["passage_id"])

    print("BM25 score:", round(passage["score"], 4))

    print("Gold evidence:", passage["is_gold_evidence"])

    print("Text:", passage["text"])
