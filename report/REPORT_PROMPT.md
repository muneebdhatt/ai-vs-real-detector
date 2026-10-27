# Prompt for generating the project report (Claude / ChatGPT)

Run training first. Then paste the prompt below into Claude or ChatGPT together with:
`report/results_baseline.json`, `report/results_mobilenet.json`, `report/history_*.json`, and your Grad-CAM notes.
The report must use only your real numbers.

---

Write a university project report (about 3,000 words) for a deep learning semester project called
"AI vs Real Image Detector". Use ONLY the numbers and facts I provide below. If something is missing,
write "[TO FILL]" instead of inventing it.

Sections: 1) Abstract, 2) Problem statement and motivation, 3) Dataset description (CIFAKE: 60k real from
CIFAR-10 + 60k Stable Diffusion images, 32x32; plus my self-collected real-world test set), 4) Preprocessing
and augmentation, 5) Model architectures (custom 4-block CNN baseline; MobileNetV2 transfer learning, head
training then fine-tuning the top 30 layers at lr 1e-5), 6) Training details (binary cross-entropy, Adam,
batch size 64, 128x128 input, early stopping), 7) Results (tables for CIFAKE test and real-world test:
accuracy, precision, recall, F1, ROC-AUC, confusion matrices), 8) Generalization gap analysis, 9) Grad-CAM
analysis, 10) Error analysis, 11) Limitations and ethical considerations, 12) Conclusion and future work,
13) References.

My results (paste JSON here):
[PASTE results_baseline.json]
[PASTE results_mobilenet.json]

My real-world test set: [N real photos from ..., N AI images from generators ...]
My Grad-CAM observations: [what regions the model focused on]
Examples the model got wrong: [describe]

---

After generating: check every number in the report against `report/results_*.json`.
