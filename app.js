let model;
let classNames = [];
const NORMALIZE_DIV255 = true;
const IMG_SIZE = 224;

let videoElement;
let currentFacing = "user";   // "user" = frontal, "environment" = trasera
let showTop5 = false;         // estado del switch Top-1 / Top-5

// --------------------------------------------------
// 1️⃣ Cargar modelo y nombres de clases
// --------------------------------------------------
async function loadModel() {
  console.log("Cargando modelo…");
  model = await tf.loadGraphModel('tfjs_model/model.json');
  console.log("✅ Modelo cargado correctamente");
  console.log("Inputs:", model.inputs);
  console.log("Outputs:", model.outputs);
}

async function loadClassNames() {
  const resp = await fetch('clases.txt');
  const text = await resp.text();
  classNames = text.split('\n').map(s => s.trim()).filter(Boolean);
  console.log(`📚 ${classNames.length} clases cargadas`);
}

// --------------------------------------------------
// 2️⃣ Cámara (inicializar y alternar)
// --------------------------------------------------
async function initCamera() {
  videoElement = document.getElementById('video');
  const constraints = { video: { facingMode: currentFacing, width: 640, height: 480 }, audio: false };

  // Cierra stream anterior si existía
  if (videoElement.srcObject) videoElement.srcObject.getTracks().forEach(t => t.stop());

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (e) {
    // fallback si "environment" no existe
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  }

  videoElement.srcObject = stream;
  await videoElement.play();
  console.log(`📷 Cámara activada (${currentFacing})`);
}

async function switchCamera() {
  currentFacing = currentFacing === "user" ? "environment" : "user";
  await initCamera();
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
// 4️⃣ Predicción (Top-1 / Top-5 con softmax)
// --------------------------------------------------
async function predictFromCamera() {
  if (!model) {
    alert("Primero carga el modelo");
    return;
  }

  const x = preprocessFromVideo();
  const inputName = model.inputs[0].name;
  let out = await model.executeAsync({ [inputName]: x });
  let y = Array.isArray(out) ? out[0] : out;   // [1, C] o [C]

  if (y.shape.length > 2) y = y.squeeze();
  if (y.shape.length === 1) y = y.expandDims(0); // [1, C]

  // Softmax y saneo
  let probsT = tf.softmax(y);
  const probs = Array.from(await probsT.data()).map(v => Number.isFinite(v) ? v : 0);

  // Top-k
  const sorted = probs.map((p,i)=>({prob:p, idx:i})).sort((a,b)=>b.prob-a.prob);
  const top = (showTop5 ? sorted.slice(0,5) : sorted.slice(0,1));

  // Render
  const html = top.map((r,i)=>{
    const name = classNames[r.idx] ?? `Clase ${r.idx}`;
    const pct  = (r.prob*100).toFixed(2);
    return `<div class="pred-item"><b>${i+1}. ${name}</b> — ${pct}%</div>`;
  }).join("");

  document.getElementById("result").innerHTML =
    `<h3>${showTop5 ? "Top-5 predicciones" : "Predicción principal"}</h3>${html}`;

  // Limpieza
  x.dispose();
  probsT.dispose();
  if (Array.isArray(out)) out.forEach(t=>t.dispose()); else out.dispose();
}

// --------------------------------------------------
// 5️⃣ Inicialización y eventos UI
// --------------------------------------------------
window.addEventListener('DOMContentLoaded', async () => {
  await loadModel();
  await loadClassNames();
  await initCamera();

  // Botón predecir
  document.getElementById('predictBtn').addEventListener('click', predictFromCamera);

  // Botón cambiar cámara
  const switchBtn = document.getElementById('switchBtn');
  if (switchBtn) switchBtn.addEventListener('click', switchCamera);

  // Switch Top-1 / Top-5
  const toggleTop = document.getElementById('toggleTop');
  const topLabel  = document.getElementById('topLabel');
  if (toggleTop) {
    toggleTop.addEventListener('change', (e)=>{
      showTop5 = e.target.checked;
      if (topLabel) topLabel.textContent = showTop5 ? "Top-5" : "Top-1";
    });
  }
});
