# Pre-specified analysis plan

Written and committed before any model was evaluated on the data. Changes after this commit are listed in the README under "Deviations".

## Question
Do image-level differences in how closely a network's evidence map matches human diagnostic-feature maps carry information about whether the network's prediction is correct, beyond what its confidence already says?

## Data and models
- ClickMe validation split (17,408 ImageNet-val images with human importance maps; Linsley et al., 2019). Fixed random subset of N = 1,500 images (seed 0), same images for every model.
- Eight torchvision ImageNet models with default pretrained weights: AlexNet, VGG-16, ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0, ConvNeXt-Tiny, ViT-B/16.

## Measures (per image i, per model m)
- Confidence c: maximum softmax probability. Correct y: top-1 equals the label.
- Model evidence map: |gradient of the true-class logit w.r.t. input|, max over colour channels (the lab's default saliency protocol, Fel et al. 2022).
- Human map: ClickMe heatmap, resized to 64x64, Gaussian-blurred (size 10, sigma 10), resized to 224x224 (same preprocessing as the released code).
- Alignment rho_i: Spearman correlation (average ranks for ties) between model map and human map. Normalised alignment = mean(rho)/0.65753 (human inter-rater ceiling reported by the lab).
- Null alignment: same, but the human map is taken from three other random images. Excess alignment e_i = rho_i - mean null. e is the primary image-level predictor because ClickMe maps have a centre bias.
- Anchoring index g_i: drop in true-class probability when the top 20% of human-map pixels are replaced with the mean colour, minus the drop when the top 20% of a different image's human map is masked.

## Hypotheses
- V1 (validity). Mean rho exceeds mean null alignment for every model, and across the eight models accuracy and normalised alignment are not positively related (the published trade-off, in miniature).
- H1 (model level, descriptive only; n = 8 is too small to confirm). Report Spearman between normalised alignment and ECE, and the same after residualising on accuracy.
- H2 (confirmatory). For each model, Y is independent of e given c. Test statistic: within-bin AUROC of e for y, averaged over 10 confidence-quantile bins weighted by number of positive-negative pairs. Null distribution: 5,000 permutations of e within bins. Two-sided, Holm correction over the eight models, alpha = 0.05. Effect size: change in cross-validated AUROC when e is added to a logistic model of y on logit(c).
- H3 (secondary). Same test as H2 with g in place of e.
- Descriptive: ECE (15 equal-width bins) within terciles of e, with 95% bootstrap intervals.

## Reading the result
A null on H2 means confidence already carries what alignment offers. A positive result means alignment tracks reliability beyond confidence. Both are reported.

## Not claimed
Alignment needs a human map and the true label, so it is a diagnostic probe, not a deployable confidence score. Results come from pretrained models and one saliency method; no claim is made about training for alignment.
