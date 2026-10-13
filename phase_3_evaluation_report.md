# Phase 3 Evaluation Report: Baseline vs. Augmented Models

This report details the outcomes of **Phase 1 (Baseline Reproduction)** and **Phase 2 (Realistic Augmentation)**, evaluated consistently on the test set and subjected to the background-intervention (masking) test. 

All phases were executed sequentially without modifying existing checkpoints, utilizing the identical validation-based early stopping logic.

## 1. Experiment A: Baseline Reproduction
**Goal:** Verify the exact reproducibility of the previously audited ResNet18 baseline configuration.
*   **Training Script:** `train_expA_baseline.py`
*   **Checkpoint Path:** `checkpoints/expA_baseline_best.pth`
*   **Seed:** 42
*   **Training Configuration:** ResNet18 (ImageNet weights), Adam optimizer (LR Backbone 5e-5, LR Head 1e-3), 8-epoch patience.

### Test Set Performance (Exp A)
*   **Accuracy:** 88.57% (62/70)
*   **Macro F1:** 0.8384
*   **Weighted F1:** 0.8750
*   **Per-Class Metrics:**
    *   **Healthy:** Precision 1.000, Recall 0.909, F1 0.952
    *   **Magnesium:** Precision 0.905, Recall 1.000, F1 0.950
    *   **Nitrogen:** Precision 0.778, Recall 0.933, F1 0.848
    *   **Phosphorus:** Precision 0.941, Recall 0.941, F1 0.941
    *   **Potassium:** Precision 0.750, Recall 0.375, F1 0.500

### Background Intervention Sensitivity (Exp A)
When the background was replaced with a neutral ImageNet mean via GrabCut masking:
*   **Correct → Incorrect Flips:** **44**

*Conclusion for Exp A:* The baseline was **perfectly reproduced**. The metrics exactly match the initial step's report, right down to the 44 background flips, proving that the training pipeline and the background vulnerability are fully deterministic and reproducible.

---

## 2. Experiment B: Enhanced Realistic Augmentation
**Goal:** Test if adding moderate geometric (`RandomAffine`) and broader lighting augmentations (`brightness=0.25, contrast=0.25`) strictly during training mitigates background reliance.
*   **Training Script:** `train_expB_augmented.py`
*   **Checkpoint Path:** `checkpoints/expB_augmented_best.pth`
*   **Seed:** 42
*   **Training Configuration:** Identical to baseline, except for the enhanced `train_transform`. Validation/test transforms remained strictly deterministic.

### Test Set Performance (Exp B)
*   **Accuracy:** 85.71% (60/70)
*   **Macro F1:** 0.8333
*   **Weighted F1:** 0.8519
*   **Per-Class Metrics:**
    *   **Healthy:** Precision 0.917, Recall 1.000, F1 0.957
    *   **Magnesium:** Precision 0.941, Recall 0.842, F1 0.889
    *   **Nitrogen:** Precision 0.778, Recall 0.933, F1 0.848
    *   **Phosphorus:** Precision 0.833, Recall 0.882, F1 0.857
    *   **Potassium:** Precision 0.800, Recall 0.500, F1 0.615

### Background Intervention Sensitivity (Exp B)
When the background was replaced with a neutral ImageNet mean via GrabCut masking:
*   **Correct → Incorrect Flips:** **43**

*Conclusion for Exp B:* The enhanced realistic augmentation policy slightly harmed overall test accuracy (dropping from 88.6% to 85.7%) but did slightly improve Potassium recall (from 37.5% to 50%). **Crucially, it entirely failed to solve the background dependence issue.** The model still suffered 43 correct-to-incorrect flips upon background removal, virtually identical to the baseline (44 flips). 

---

## 3. Confusion Matrix and Predictions
*   The raw test predictions and confusion matrices were successfully exported for both models. 
*   **Exp A Matrix:** `results/ExpA_Baseline_confusion_matrix.png`
*   **Exp B Matrix:** `results/ExpB_Augmented_confusion_matrix.png`

Both models continue to struggle heavily with `Potassium`, frequently misclassifying it, which correlates with Potassium's distinct marginal necrosis symptoms that might be underrepresented or easily confused with geometric distortions.

## 4. Summary & Next Steps
We have conclusively proven that:
1.  The baseline is reproducible.
2.  Standard, realistic image augmentations (Phase 2) are **insufficient** to break the network's reliance on the static background correlations present in the training set.

We are now ready to proceed to **Phase 4: Background-Masking Experiment**, pending your review. As verified previously, the GrabCut masks effectively preserve leaf tissue in the vast majority of cases, meaning an active masking intervention during training is a viable next step to force the network to learn actual leaf features.
