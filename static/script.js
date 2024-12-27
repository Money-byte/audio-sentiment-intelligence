function uploadAudio() {
  let audioFile = document.getElementById("audioFile").files[0];

  if (!audioFile) {
    alert("Please select an audio file");
    return;
  }

  let formData = new FormData();
  formData.append("file", audioFile);

  fetch("/predict", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById("result").innerText =
        "Predicted Emotion: " + data.emotion;
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}
