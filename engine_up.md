Strategy: From Simple Search to Intent-Driven Narrative
The core issue identified—the model dropping relevant clips—stems from a mismatch between the user's intent and the model's instructions. The current "one-size-fits-all" prompt instructs the AI to be a ruthless editor, which fails when the user's goal is comprehensive overview or example gathering.

This document outlines a more sophisticated, two-stage approach:

Intent Classification: First, understand what kind of answer the user is looking for.

Intent-Specific Prompting: Then, provide the AI with a tailored set of instructions that matches that specific intent.

1. Deconstructing User Intent: What is the User Really Asking?
Not all queries are created equal. We can classify the majority of user queries into four primary categories of intent. By identifying the intent first, we can dramatically improve the quality of the final video.

Intent Category

Description

Example Queries

Ideal Output

Exploratory

The user wants a general, comprehensive overview of a broad topic. They are exploring, not pinpointing.

"What did he say about the economy?" <br> "Tell me about his foreign policy."

A 2-3 minute supercut that includes a representative sample of the most important statements on the topic.

Specific / Factual

The user is looking for a single, precise moment or a definitive statement. They are trying to locate a specific fact.

"Find the part where she announces the new product." <br> "What were the exact Q3 revenue numbers?"

A short, focused clip of the single most relevant segment, potentially with the preceding and succeeding sentence for context.

Exemplary

The user wants a collection of examples that fit a certain theme or mood. The goal is quantity and variety over a single narrative.

"Show me some funny moments." <br> "Find clips of audience reactions."

A montage of several short, distinct clips that all fit the requested theme.

Comparative

The user wants to compare and contrast statements, often to track changes over time or identify contradictions.

"How has his view on climate change evolved?" <br> "Compare his statements on immigration from the start and end of the talk."

A sequence of 2-3 clips, explicitly chosen from different points in time, that highlight the comparison or contrast.

2. The New Algorithm: An Intent-Driven Workflow
We will evolve the backend from a single-step process to a more intelligent, multi-step workflow.

Initial Semantic Search (As Before): The system first retrieves a broad set of semantically relevant clips (e.g., the top 20 matches). This part of the process remains unchanged.

NEW - Intent Classification: The user's query is sent to the LLM with a new, simple "classifier" prompt:

"Classify the following user query into one of four categories: Exploratory, Specific, Exemplary, or Comparative. Respond with only the category name. Query: '{user_query}'"

Dynamic Prompt Selection: Based on the classification from the previous step, the system selects one of the new, intent-specific prompts (detailed below).

Narrative Generation (As Before): The selected prompt, along with the relevant clips, is sent to the LLM to generate the final JSON playlist.

This pre-processing step is extremely fast and ensures that the main narrative generation task is guided by a much more context-aware set of instructions.

3. Advanced Prompts: Tailoring Instructions to the Intent
Here are the new, specialized prompts designed for each intent.

A. For "Exploratory" Queries
This prompt encourages comprehensiveness over aggressive editing.

You are an expert documentary film editor. Your task is to create a comprehensive 2-minute overview that answers the user's broad question: "{query}"

You have been given a collection of relevant video clips. Your goal is to select a representative sample that covers the main points of the topic.

RULES:
1. Create a logical narrative flow.
2. **Prioritize including a variety of key moments over being overly concise.** It is better to include a slightly less relevant clip than to leave out a key aspect of the topic.
3. The output MUST be a valid JSON array of objects. Each object must have "start", "end", and "narrative_reason" keys.

Here are the available clips:
{json.dumps(relevant_segments, indent=2)}

Generate the final JSON script:

B. For "Specific / Factual" Queries
This prompt instructs the model to be precise and add context.

You are a research assistant. Your task is to find the single, most definitive video clip that answers the user's specific question: "{query}"

You have been given a ranked list of the most relevant clips. Your job is to select the ONE best clip.

RULES:
1. **You MUST select only one clip from the list.**
2. Choose the clip that most directly and factually answers the user's question.
3. In the "narrative_reason", briefly explain why this clip is the most direct answer.
4. The output MUST be a valid JSON array containing a single object.

Here are the available clips:
{json.dumps(relevant_segments, indent=2)}

Generate the final JSON script containing only the single best clip:

C. For "Exemplary" Queries
This prompt focuses on gathering a collection of moments.

You are a video producer creating a montage. Your goal is to find several good examples that match the user's request for: "{query}"

You have been given a collection of relevant video clips.

RULES:
1. Select 3 to 5 of the best clips that fit the theme.
2. Do not worry about creating a single, overarching narrative. The goal is to present a collection of strong, individual moments.
3. The output MUST be a valid JSON array of objects.

Here are the available clips:
{json.dumps(relevant_segments, indent=2)}

Generate the final JSON script for the montage:

Conclusion
By implementing this intent-driven framework, we transform the Narrative Engine from a simple summarizer into a context-aware content creation tool. This approach directly solves the problem of the AI being too aggressive in discarding clips for broad queries and provides a more robust and reliable foundation for generating high-quality, personalized video content that truly matches what the user wants to see.

---------------------------------------------------------------------------------------------------------

