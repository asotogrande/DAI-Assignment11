# Part 1 — NER + Knowledge Graph

**Date:** 2025-11-17

## Data & Method
- Parsed sections from `full_info` (Corporate/Academic/Background), ran NER (Hugging Face), normalized aliases, and built a typed NetworkX graph (professors ↔ entities).

## Graph Overview
- Nodes: 925 | Edges: 1108
- Connected components: 2
- Largest component: 923 nodes (99.8% of all nodes)

## Highlights
- Top universities (by degree/betweenness): Ie University Spain, Ie School Of Business, Universidad Politécnica De Madrid
- Top companies: A, Legal Department, S
- Bridge nodes (betweenness): Spain, Appius Licinius Cicero, Horatia Pulchra

## Visual
- Betweenness Top-K saved as `plot_betweenness_topK.png`.

