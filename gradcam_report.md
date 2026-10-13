# Grad-CAM Explainability Report

This report summarizes the implementation of Gradient-weighted Class Activation Mapping (Grad-CAM) to provide explainability for the trained maize leaf nutrient deficiency model.

## 1. Selected Model
Based on the validation and test evidence from the 5-model comparison, **ResNet18** was selected for Grad-CAM analysis. 
*   **Validation Evidence:** ResNet18 achieved a robust validation accuracy of 88.4% and a low validation loss of 0.3319 at epoch 9.
*   **Test Evidence:** It generalized best on the held-out test set, achieving the highest accuracy (88.57%) and the highest macro F1 score (0.8384) among all five tested algorithms.
*   **Layer Selection:** The Grad-CAM hooks were attached to the final convolutional layer of the ResNet18 backbone (`layer4[-1]`), capturing the highest-level spatial features before global pooling and classification.

## 2. Method
Grad-CAM uses the gradients of any target concept (in this case, the predicted class score) flowing into the final convolutional layer to produce a coarse localization map. This heatmap highlights the important regions in the image that most strongly influenced the CNN's prediction. 

We used the deterministic test preprocessing transform (`val_transform`) to ensure that the image fed to Grad-CAM exactly matches the image evaluated during test metrics computation. The generated heatmap was then resized, colored, and overlaid onto the un-normalized original image.

## 3. Examples Inspected
We generated side-by-side (Original vs. Grad-CAM Overlay) visualizations for multiple test set images, covering correct and incorrect predictions across different classes:

1.  **Healthy (Correct)** - `Correct_Healthy_idx0.png`
2.  **Magnesium (Correct)** - `Correct_Magnesium_idx11.png`
3.  **Nitrogen (Correct)** - `Correct_Nitrogen_idx30.png`
4.  **Phosphorus (Correct)** - `Correct_Phosphorus_idx45.png`
5.  **Potassium (Correct)** - `Correct_Potassium_idx63.png`
6.  **Phosphorus (Incorrect)** - `Incorrect_Phosphorus_idx46.png`
7.  **Potassium (Incorrect)** - `Incorrect_Potassium_idx62.png`

*(Note: The test set has only 8 Potassium examples, and the model struggles with this class. We specifically targeted incorrect predictions here to understand the misclassifications).*

## 4. Observations
*   **Focused Attention:** In correct predictions (e.g., Magnesium and Nitrogen deficiency), the Grad-CAM heatmap often focuses cleanly on the leaf area showing distinct discoloration (like interveinal chlorosis or yellowing tips).
*   **Background Interference:** In some misclassified examples (like the incorrect Potassium or Phosphorus), the heatmap may show some attention drifting toward the background or the edges of the image, suggesting the model might be picking up on spurious correlations rather than leaf symptoms.
*   **Phosphorus and Potassium Limitations:** The model correctly identified some instances by focusing on leaf edges (characteristic of Potassium deficiency) and purpling (characteristic of Phosphorus). However, in the incorrect examples, the network either focuses on the wrong leaf segment or acts confused, correlating with the lower F1 scores for these classes.

## 5. Limitations
*   **Explainability vs. Causation:** Grad-CAM **only shows the regions influencing the CNN prediction**. It **does not independently validate the diagnosis or establish biological causation**. If the model relies on a background artifact (like a shadow or a piece of dirt) to make a prediction, Grad-CAM will highlight that artifact. It tells us *where* the model looked, not *why* the disease is happening biologically.
*   **Resolution:** The final convolutional layer of ResNet18 is relatively low resolution (7x7 for a 224x224 input). The interpolated heatmap is coarse and cannot pinpoint microscopic or very fine-grained symptoms (e.g., tiny necrotic spots).
*   **Pre-trained Bias:** Since ResNet18 is pre-trained on ImageNet, its feature maps are biased towards object edges and textures present in natural images. This may influence which patterns the network "prefers" to focus on during fine-tuning.
