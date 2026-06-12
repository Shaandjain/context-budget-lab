# Dataset Slice

JSONL datasets for the context-budget benchmark loop. The v0 suites are still
small, but they are no longer toy smoke fixtures: they exercise citation QA and
synthetic memory recall with explicit distractors, expected facts, and output
schema names.

## Schema

Each line is one task object:

```json
{
  "task_id": "stable-task-id",
  "dataset_id": "public_ai_policy_v0",
  "task_type": "citation_qa",
  "query": "Question shown to the model.",
  "context": [
    {
      "document_id": "stable-document-id",
      "title": "Human-readable title",
      "text": "Task-local context snippet.",
      "source_url": "https://public-source.example/path"
    }
  ],
  "expected_answer": "Short reference answer.",
  "expected_citations": ["stable-document-id"],
  "expected_facts": ["literal substring"],
  "answer_schema": "answer_with_citations",
  "metadata": {
    "source_type": "public",
    "source_urls": ["https://public-source.example/path"],
    "difficulty": "medium",
    "tags": ["citation_accuracy"],
    "distractor_document_ids": ["nearby-but-wrong-document-id"]
  }
}
```

Required fields are validated by `context_budget_lab.datasets`.

## Datasets

- `public_ai_policy_v0.jsonl`: 20 self-written public AI policy tasks using
  government, OECD, and model-card sources. Use this to exercise citation
  selection across similar policy/governance passages.
- `synthetic_agent_memory_v0.jsonl`: 20 synthetic agent-memory tasks inspired by
  public agent/retrieval papers. Use this to test preference recall, state
  recall, conflict resolution, and explicit "memory does not contain this"
  abstention without private data.

## Source Rules

- Public datasets include the public source URL on each context document and in
  `metadata.source_urls`.
- Synthetic datasets label `metadata.source_type` as `synthetic` and include
  public method/source URLs in `metadata.source_urls`; the task content itself
  is invented.
- Do not add Casper client data here.
