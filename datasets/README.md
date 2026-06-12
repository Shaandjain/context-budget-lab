# Dataset Slice

Tiny JSONL datasets for the first context-budget benchmark loop. These are not
claim-making evaluation sets; they are smoke-test fixtures for loader,
prompting, trace emission, and scoring integration.

## Schema

Each line is one task object:

```json
{
  "task_id": "stable-task-id",
  "dataset_id": "public_ai_policy_toy",
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
  "metadata": {
    "source_type": "public",
    "source_urls": ["https://public-source.example/path"],
    "benchmark_focus": ["citation_accuracy"],
    "difficulty": "toy"
  }
}
```

Required fields are validated by `context_budget_lab.datasets`.

## Datasets

- `public_ai_policy_toy.jsonl`: public AI policy snippets from government or
  official public sources. Use this to exercise citation selection across
  similar policy/governance passages.
- `synthetic_agent_memory_toy.jsonl`: synthetic agent-memory tasks inspired by
  public agent/retrieval papers. Use this to test memory compression and
  preference/state recall without private data.

## Source Rules

- Public datasets include the public source URL on each context document and in
  `metadata.source_urls`.
- Synthetic datasets label `metadata.source_type` as `synthetic` and include
  public method/source URLs in `metadata.source_urls`; the task content itself
  is invented.
- Do not add Casper client data here.
