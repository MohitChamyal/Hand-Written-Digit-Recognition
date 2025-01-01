let model;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const clearButton = document.getElementById('clear-btn');
const predictButton = document.getElementById('predict-btn');
const predictionDisplay = document.getElementById('prediction');

async function loadModel() {
  model = await tf.loadLayersModel('path/to/tfjs_model/model.json');
  console.log("Model loaded");
}

loadModel();

// Draw on the canvas
let drawing = false;
canvas.addEventListener('mousedown', () => drawing = true);
canvas.addEventListener('mouseup', () => drawing = false);
canvas.addEventListener('mousemove', draw);

function draw(event) {
  if (!drawing) return;
  ctx.lineWidth = 15;
  ctx.lineCap = 'round';
  ctx.strokeStyle = '#000';
  ctx.lineTo(event.offsetX, event.offsetY);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(event.offsetX, event.offsetY);
}

// Clear the canvas
clearButton.addEventListener('click', () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  predictionDisplay.textContent = '-';
});

// Predict digit
predictButton.addEventListener('click', async () => {
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const tensor = tf.browser.fromPixels(imageData, 1)
                  .resizeNearestNeighbor([28, 28])
                  .mean(2)
                  .toFloat()
                  .expandDims(0)
                  .expandDims(-1)
                  .div(tf.scalar(255));
  
  const prediction = await model.predict(tensor).data();
  const predictedDigit = prediction.indexOf(Math.max(...prediction));
  
  predictionDisplay.textContent = predictedDigit;
});
