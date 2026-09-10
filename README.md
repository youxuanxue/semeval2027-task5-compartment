# SemEval-2027 Task 5: ComPartMent
**Noun Compound and Particle Verb Compositionality Prediction in Context**

Official Codabench Competition ID: [17971](https://www.codabench.org/competitions/17971/)  
Organizers: Filip Miletić, Chris Jenkins, Sabine Schulte im Walde (University of Stuttgart)

## Tasks
- **Subtask A (Noun Compounds)**: Produce rankings for modifier and head compositionality in context.
- **Subtask B (Particle Verbs)**: Produce rankings for particle verb compositionality in context.
- Languages: English (`en`) and German (`de`).
- Time periods: Mixed historical and present-day texts.

## Dataset Splits
- `data/train/en-nn-train.tsv`: English Noun Compounds (444 targets, 3,480 sentences)
- `data/train/en-pv-train.tsv`: English Particle Verbs (158 targets, 1,557 sentences)
- `data/train/de-nn-train.tsv`: German Noun Compounds (461 targets, 3,298 sentences)
- `data/train/de-pv-train.tsv`: German Particle Verbs (158 targets, 1,490 sentences)

## Evaluation Metrics
- Primary: Spearman's rank correlation (rho)
- Secondary: Context variance correlation (rho) and RMSE
