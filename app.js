let model;
let classNames = [];
const NORMALIZE_DIV255 = true;
const IMG_SIZE = 224;
let videoElement;

// --------------------------------------------------
// 1️⃣ Cargar modelo y nombres de clases
// --------------------------------------------------
async function loadModel() {
  console.log("Cargando modelo...");
  model = await tf.loadGraphModel('tfjs_model/model.json');
  console.log("✅ Modelo cargado correctamente");
  console.log("Inputs del modelo:", model.inputs);
  console.log("Outputs del modelo:", model.outputs);
}

async function loadClassNames() {
  const response = await fetch('clases.txt');
  const text = await response.text();
  // cada línea es una clase, las limpiamos y guardamos
  classNames = text.split('\n').map(l => l.trim()).filter(l => l.length > 0);
  console.log(`📚 ${classNames.length} clases cargadas`);
}

// --------------------------------------------------
// 2️⃣ Activar cámara
// --------------------------------------------------
async function initCamera() {
  videoElement = document.getElementById('video');
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: 640, height: 480 },
    audio: false
  });
  videoElement.srcObject = stream;
  await videoElement.play();
  console.log("📷 Cámara activada");
}

// --------------------------------------------------
// 3️⃣ Preprocesar frame
// --------------------------------------------------
function preprocessFromVideo() {
  return tf.tidy(() => {
    let img = tf.browser.fromPixels(videoElement).toFloat();
    img = tf.image.resizeBilinear(img, [IMG_SIZE, IMG_SIZE]);
    if (NORMALIZE_DIV255) img = img.div(255.0);
    return img.expandDims(0); // [1,224,224,3]
  });
}

// --------------------------------------------------
// 4️⃣ Ejecutar predicción
// --------------------------------------------------
async function predictFromCamera() {
  if (!model) {
    alert("Primero carga el modelo");
    return;
  }

  const inputTensor = preprocessFromVideo();
  const inputName = model.inputs[0].name;
  const output = await model.executeAsync({ [inputName]: inputTensor });
  const prediction = Array.isArray(output) ? output[0] : output;

  const probs = await prediction.data();
  const maxIdx = probs.indexOf(Math.max(...probs));
  const className = classNames[maxIdx] || `Clase ${maxIdx}`;

  document.getElementById("result").innerText = `Predicción: ${className}`;

  // limpieza de tensores
  inputTensor.dispose();
  prediction.dispose();
  if (Array.isArray(output)) output.forEach(t => t.dispose());
}

// --------------------------------------------------
// 5️⃣ Inicialización
// --------------------------------------------------
window.addEventListener('DOMContentLoaded', async () => {
  await initCamera();
  await loadModel();
  await loadClassNames();

  document.getElementById('predictBtn').addEventListener('click', predictFromCamera);
});
