// Edit this file after you have your real links and numbers.
// Replace every "PUT_..._HERE" value before submitting.
window.PROJECT_CONFIG = {
  modelUrl: "model/model.json",
  imgSize: 128,
  uncertainBelow: 0.65,   // show "Uncertain" when confidence is below this

  links: [
    { id: "github",  label: "GitHub repository",   hint: "Code, notebooks, model weights",  url: "https://github.com/muneebdhatt/ai-vs-real-detector" },
    { id: "dataset", label: "Dataset (CIFAKE)",    hint: "Kaggle dataset",                  url: "https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images" },
    { id: "report",  label: "Project report",      hint: "Full article / report",           url: "PUT_REPORT_LINK_HERE" },
    { id: "slides",  label: "Presentation (PPT)",  hint: "Slides",                          url: "PUT_PRESENTATION_LINK_HERE" },
    { id: "video",   label: "YouTube video",       hint: "Voiceover walkthrough",           url: "PUT_YOUTUBE_LINK_HERE" },
    { id: "linkedin",label: "LinkedIn post",       hint: "Project announcement",            url: "PUT_LINKEDIN_POST_LINK_HERE" }
  ],

  // Fill from report/results_baseline.json and report/results_mobilenet.json
  results: [
    { model: "Custom CNN (CIFAKE only)",            cifake: "95.3%", realworld: "50.0%" },
    { model: "MobileNetV2 (CIFAKE only)",           cifake: "94.7%", realworld: "50.0%" },
    { model: "MobileNetV2 (mixed data, deployed)",  cifake: "90.4%", realworld: "86.3%" }
  ]
};
