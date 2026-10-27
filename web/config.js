// Edit this file after you have your real links and numbers.
// Replace every "PUT_..._HERE" value before submitting.
window.PROJECT_CONFIG = {
  modelUrl: "model/model.json",
  imgSize: 128,
  uncertainBelow: 0.65,   // show "Uncertain" when confidence is below this

  links: [
    { id: "github",  label: "GitHub repository",   hint: "Code, notebooks, model weights",  url: "PUT_GITHUB_REPO_URL_HERE" },
    { id: "dataset", label: "Dataset (CIFAKE)",    hint: "Kaggle dataset",                  url: "https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images" },
    { id: "report",  label: "Project report",      hint: "Full article / report",           url: "PUT_REPORT_LINK_HERE" },
    { id: "slides",  label: "Presentation (PPT)",  hint: "Slides",                          url: "PUT_PRESENTATION_LINK_HERE" },
    { id: "video",   label: "YouTube video",       hint: "Voiceover walkthrough",           url: "PUT_YOUTUBE_LINK_HERE" },
    { id: "linkedin",label: "LinkedIn post",       hint: "Project announcement",            url: "PUT_LINKEDIN_POST_LINK_HERE" }
  ],

  // Fill from report/results_baseline.json and report/results_mobilenet.json
  results: [
    // { model: "Custom CNN (baseline)", cifake: "00.0%", realworld: "00.0%" },
    // { model: "MobileNetV2 (transfer learning)", cifake: "00.0%", realworld: "00.0%" }
  ]
};
