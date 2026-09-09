# FINAL PRIMARY RESULT DECISION

## Question
*"Can Event F1 = 0.604 be used as the primary quantitative result in the dissertation?"*

## Answer
**YES.**

## Justification
The real-to-real evaluation (Event F1 = 0.604) is scientifically sound, leakage-free, and represents a genuine test of the pipeline on unseen professional football tracking data. It corrects the critical flaw of the synthetic evaluation (which relied on engineered, deterministic pressing signatures). The validation and test sets were chronologically separated by a strict 100-frame purge, and all post-processing thresholds were frozen prior to test evaluation.

## Recommended Dissertation Wording
When reporting the final system performance, the dissertation should use the following framing:

> "The proposed pipeline was evaluated in a strict chronological holdout experiment on professional optical tracking data (Metrica Match 2 and Match 3). Training was conducted entirely on Match 2, while post-processing configurations—including a 3.0-second causal temporal smoothing window and a decision threshold of 0.15—were selected exclusively on the first 70% of Match 3. 
> 
> When evaluated on the completely unseen final 30% of Match 3, the model achieved a primary **Event F1 of 0.604** (95% CI: [0.490, 0.702]), with an Event Precision of 0.533 and Event Recall of 0.696. The ROC-AUC on the held-out test frames was 0.837.
> 
> This real-to-real result demonstrates that despite the failure of models trained on synthetic kinematic simulations to transfer across domains, the underlying spatial and kinematic feature set successfully captures genuine turnover-associated press-trigger dynamics when trained on authentic tracking distributions."

## Caveats to explicitly include in the Discussion chapter:
1. **Target Formulation Proxy:** The performance evaluates the model against an operational proxy target (the 3.0m proximity rule), which remains a supervised approximation of pressing, not ground-truth pressing intent.
2. **Within-Match Validation Design:** The validation and test sets were drawn from the same match (Match 3) using a chronological split. While the 100-frame purge prevented mathematical boundary leakage, the test data may share latent match-specific tactical patterns with the validation data. True leave-one-match-out external validation remains an area for future work as larger datasets become available.
3. **Evaluator Precision Logic:** The evaluator mathematically calculates precision using the number of Ground Truth events matched, which slightly penalises fragmented predictions.
