"""Prompt templates for the PatternBloom multi-turn reasoning agent."""

from __future__ import annotations


STAGE1_PROMPT = """Answer the given question by building an explicit reasoning path through a knowledge graph.

You must first reason inside <think>...</think>. Then use these actions:

1. Retrieve knowledge about a specific entity:
<query>entity_name</query>

2. After receiving knowledge, extract evidence triples and continue searching:
<graph>EntityA | relation | EntityB</graph>
<query>next_entity</query>

3. When you have sufficient evidence, provide the final answer:
<answer>your answer</answer>

First turn:
<think>
...
</think>
<query>entity_name</query>

After receiving knowledge (extract evidence AND continue searching):
<think>
...
</think>
<graph>EntityA | relation | EntityB</graph>
<query>next_entity</query>

When you have enough evidence (extract final evidence AND answer):
<think>
...
</think>
<graph>EntityA | relation | EntityB</graph>
<answer>
...
</answer>
"""


STAGE2_PROMPT = """{knowledge_block}

Answer the given question by building an explicit reasoning path through a knowledge graph.

Use the retrieved reasoning patterns above as a guide for which entities to query and which relations to extract, but adapt them to the specific question. The patterns are hints, not strict templates.

You must first reason inside <think>...</think>. Then use these actions:

1. Retrieve knowledge about a specific entity:
<query>entity_name</query>

2. After receiving knowledge, extract evidence triples and continue searching:
<graph>EntityA | relation | EntityB</graph>
<query>next_entity</query>

3. When you have sufficient evidence, provide the final answer:
<answer>your answer</answer>

First turn:
<think>
...
</think>
<query>entity_name</query>

After receiving knowledge (extract evidence AND continue searching):
<think>
...
</think>
<graph>EntityA | relation | EntityB</graph>
<query>next_entity</query>

When you have enough evidence (extract final evidence AND answer):
<think>
...
</think>
<graph>EntityA | relation | EntityB</graph>
<answer>
...
</answer>
"""


KNOWLEDGE_BLOCK_TEMPLATE = "<knowledge>\n{patterns}\n</knowledge>"


def build_stage1_user_message(question: str) -> str:
    return STAGE1_PROMPT + "Question: " + question


def build_stage2_user_message(question: str, pattern_hint: str) -> str:
    knowledge_block = KNOWLEDGE_BLOCK_TEMPLATE.format(
        patterns=pattern_hint.strip()
    )
    return STAGE2_PROMPT.format(knowledge_block=knowledge_block) + "Question: " + question
