# Research Contract: Novel Lightweight PII Detection Model Architecture

## Classification: Type C (Analysis)
- Requires judgment across multiple technical domains
- Multiple architectural perspectives to evaluate
- Novel design decisions required

## Core Question
What is the optimal architecture for a PII-specific detection model that is dramatically smaller (<50M params), faster, and more accurate on PII than general-purpose models like GLiNER2 (205M params)?

## Use Case
Inform the design and implementation of DataFog's next-generation PII detection model — a novel, research-grade contribution to the field.

## Audience
Technical — ML engineers building the model

## Scope
- **Inclusions**: All NER architectures relevant to PII, PII-specific datasets, evaluation frameworks, efficient model design, character-level and contextual approaches
- **Exclusions**: General LLM-based approaches (too large), non-English-first approaches, vision-based PII detection
- **Timeframe**: Focus on 2020-2025 literature, with key foundational work from earlier

## Working Hypothesis
A hybrid character-contextual architecture that jointly learns character-level patterns (structured PII) and contextual representations (soft PII) in a single differentiable model can achieve SOTA PII detection at <50M parameters.

## Output Format
Full research report with executive summary, findings, recommendations, and evidence ledger

## Citation Level
Standard — academic sources preferred, GitHub repos and technical blogs acceptable for implementation details

## Research Subquestions
1. What architectures exist for PII/NER detection and what are their tradeoffs?
2. What PII-specific training datasets and benchmarks exist?
3. How should PII detection be evaluated differently from general NER?
4. What is the design space for our hybrid architecture?
5. Where are the genuine gaps in the literature for novel contribution?

## Hypotheses to Test
- H1: A PII-specific model can achieve higher PII F1 than GLiNER2 at <25% of its parameter count
- H2: Joint character-contextual learning outperforms pipeline approaches (regex→NER) for PII
- H3: PII-specific pretraining/data curation is more impactful than architecture choice alone
- H4: Span-based approaches outperform token classification for PII entity types
- H5: A model under 30M params can match or exceed 205M GLiNER2 on PII benchmarks
