# Temporal Relevance Scenario Generation

Generate training scenarios that test temporal awareness in memory retrieval.

## Instructions

Generate {batch_size} scenarios. Each scenario has:

1. A query with implicit temporal context
2. A set of 3-5 candidate memories with different timestamps and access patterns
3. A relevance ranking that accounts for temporal factors

## Temporal Factors

- **Recency**: More recent memories are generally more relevant
- **Access frequency**: Frequently accessed memories are likely still important
- **Memory type**: Feedback memories tend to be more durable; project memories decay faster
- **Staleness signals**: Project memories about completed work become less relevant

## Output Format (JSONL)

```json
{
    "query": "what is the current deployment process?",
    "now": "2026-03-27T10:00:00Z",
    "candidates": [
        {
            "type": "project",
            "description": "Deployment now uses GitHub Actions with auto-rollback",
            "created_at": "2026-03-20T09:00:00Z",
            "last_accessed_at": "2026-03-26T15:00:00Z",
            "access_count": 8,
            "temporal_relevance": 0.95,
            "reason": "Recent, frequently accessed, directly answers the query"
        },
        {
            "type": "project",
            "description": "We switched from Jenkins to GitHub Actions last month",
            "created_at": "2026-02-15T11:00:00Z",
            "last_accessed_at": "2026-02-20T14:00:00Z",
            "access_count": 3,
            "temporal_relevance": 0.5,
            "reason": "Relevant context but older and not recently accessed — may be superseded"
        },
        {
            "type": "feedback",
            "description": "Always run smoke tests after deployment",
            "created_at": "2025-12-01T10:00:00Z",
            "last_accessed_at": "2026-03-25T16:00:00Z",
            "access_count": 25,
            "temporal_relevance": 0.8,
            "reason": "Old but feedback type (durable), very frequently accessed, still actionable"
        },
        {
            "type": "project",
            "description": "Jenkins pipeline config is in /ci/Jenkinsfile",
            "created_at": "2025-08-10T09:00:00Z",
            "last_accessed_at": "2025-11-30T10:00:00Z",
            "access_count": 2,
            "temporal_relevance": 0.1,
            "reason": "Stale — refers to deprecated Jenkins setup, not accessed in months"
        }
    ]
}
```

## Scenario Types

- **Current state queries**: "what is the current X?" — favor recent memories
- **Historical queries**: "why did we decide to X?" — favor the decision memory regardless of age
- **Durable guidance**: "how should I handle X?" — feedback memories stay relevant longer
- **Expired context**: queries where the most relevant memory is NOT the most recent one
- **Frequency signals**: a rarely-accessed old memory vs a frequently-accessed old memory

## Distribution

- ~30% scenarios where recency is the primary signal
- ~25% scenarios where access frequency overrides recency
- ~25% scenarios where memory type durability matters
- ~20% edge cases (stale project memories, recently created but irrelevant, etc.)
