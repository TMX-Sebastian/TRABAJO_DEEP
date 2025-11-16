// ----------------------------
// Configuración y estado
// ----------------------------

const PLATE_MODEL_PATH = "tfjs_models/clasificador_platos/model.json";
const ING_MODEL_PATH   = "tfjs_models/segmentador_ingredientes/model.json";

const PLATE_IMG_SIZE = 224;
const ING_IMG_SIZE        = 640;
const YOLO_OBJ_THRESHOLD  = 0.10;  
const YOLO_SCORE_THRESHOLD = 0.20; 
const YOLO_MAX_BOXES      = 7;    

const NORMALIZE_DIV255 = true;

let plateModel;
let ingModel;

let plateClasses = [];        // nombres de platos (clases.txt)
let ingredientClasses = [];   // nombres de ingredientes (clases_ingredientes.txt)

let videoElement;
let snapshotCanvas;
let snapshotCtx;

let currentFacing = "environment";  // trasera por defecto
let showTop5 = false;

// ----------------------------
// Utilidades para cargar
// ----------------------------

async function loadPlateModel() {
  console.log("Cargando modelo de platos…");
  plateModel = await tf.loadGraphModel(PLATE_MODEL_PATH);
  console.log("✅ Modelo platos cargado");
}

async function loadIngModel() {
  console.log("Cargando modelo de ingredientes (YOLO)…");
  ingModel = await tf.loadGraphModel(ING_MODEL_PATH);
  console.log("✅ Modelo ingredientes cargado");
}

async function loadPlateClasses() {
  const resp = await fetch("clases.txt");
  const text = await resp.text();
  plateClasses = text.split("\n").map(s => s.trim()).filter(Boolean);
  console.log(`📚 ${plateClasses.length} clases de platos`);
}

async function loadIngredientClasses() {
  try {
    const resp = await fetch("clases_ingredientes.txt");
    const text = await resp.text();
    ingredientClasses = text.split("\n").map(s => s.trim()).filter(Boolean);
    console.log(`🥦 ${ingredientClasses.length} clases de ingredientes`);
  } catch (err) {
    console.warn("No se pudo leer clases_ingredientes.txt, se usarán índices numéricos", err);
    ingredientClasses = [];
  }
}

// ----------------------------
// Cámara
// ----------------------------

async function initCamera() {
  videoElement = document.getElementById("video");
  snapshotCanvas = document.getElementById("snapshotCanvas");
  snapshotCtx = snapshotCanvas.getContext("2d");

  const constraints = {
    video: { facingMode: currentFacing, width: 640, height: 480 },
    audio: false
  };

  if (videoElement.srcObject) {
    videoElement.srcObject.getTracks().forEach(t => t.stop());
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (e) {
    // fallback genérico
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

// ----------------------------
// Captura del frame a canvas
// ----------------------------

function captureFrameToCanvas() {
  if (!videoElement || videoElement.readyState < 2) {
    console.warn("Video aún no listo para capturar");
    return null;
  }

  const vw = videoElement.videoWidth;
  const vh = videoElement.videoHeight;
  if (!vw || !vh) {
    console.warn("Dimensiones de video no válidas");
    return null;
  }

  snapshotCanvas.width = vw;
  snapshotCanvas.height = vh;
  snapshotCtx.drawImage(videoElement, 0, 0, vw, vh);

  return { width: vw, height: vh };
}

// ----------------------------
// Preprocesamiento
// ----------------------------

function makeTensorForPlate(imgTensor) {
  let t = tf.image.resizeBilinear(imgTensor, [PLATE_IMG_SIZE, PLATE_IMG_SIZE]);
  if (NORMALIZE_DIV255) t = t.div(255.0);
  return t.expandDims(0); // [1,H,W,3]
}

function makeTensorForYolo(imgTensor) {
  let t = tf.image.resizeBilinear(imgTensor, [ING_IMG_SIZE, ING_IMG_SIZE]);
  if (NORMALIZE_DIV255) t = t.div(255.0);
  return t.expandDims(0); // [1,H,W,3]
}

// ----------------------------
// Clasificación de plato
// ----------------------------

async function runPlatePrediction(inputTensor) {
  const inputName = plateModel.inputs[0].name;
  let out = await plateModel.executeAsync({ [inputName]: inputTensor });

  let y = Array.isArray(out) ? out[0] : out;
  if (y.shape.length > 2) y = y.squeeze();
  if (y.shape.length === 1) y = y.expandDims(0); // [1, C]

  // Softmax para tener una distribución, aunque no mostremos %
  const probsT = tf.softmax(y);
  const probs = Array.from(await probsT.data()).map(v => Number.isFinite(v) ? v : 0);

  const sorted = probs
    .map((p, i) => ({ prob: p, idx: i }))
    .sort((a, b) => b.prob - a.prob);

  const top = showTop5 ? sorted.slice(0, 5) : sorted.slice(0, 1);

  const html = top.map((r, i) => {
    const name = plateClasses[r.idx] ?? `Clase ${r.idx}`;
    return `<div class="pred-item"><b>${i + 1}. ${name}</b></div>`;
  }).join("");

  const plateDiv = document.getElementById("plateResult");
  plateDiv.classList.remove("muted");
  plateDiv.innerHTML =
    `<div>${showTop5 ? "Top-5 predicciones" : "Predicción principal"}</div>${html}`;

  probsT.dispose();
  if (Array.isArray(out)) out.forEach(t => t.dispose()); else out.dispose();
}


// ----------------------------
// YOLO ingredientes – postproceso simple
// ----------------------------
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function iou(boxA, boxB) {
  const x1 = Math.max(boxA.x1, boxB.x1);
  const y1 = Math.max(boxA.y1, boxB.y1);
  const x2 = Math.min(boxA.x2, boxB.x2);
  const y2 = Math.min(boxA.y2, boxB.y2);

  const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (interArea === 0) return 0;

  const areaA = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1);
  const areaB = (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1);

  return interArea / (areaA + areaB - interArea);
}

function nonMaxSuppression(boxes, iouThreshold = 0.45) {
  const sorted = boxes.slice().sort((a, b) => b.score - a.score);
  const selected = [];

  while (sorted.length) {
    const candidate = sorted.shift();
    selected.push(candidate);

    for (let i = sorted.length - 1; i >= 0; i--) {
      if (iou(candidate, sorted[i]) > iouThreshold) {
        sorted.splice(i, 1);
      }
    }
  }
  return selected;
}

async function runIngredientsDetection(inputTensor, origWidth, origHeight) {
  if (!ingModel) return;

  const inputName = ingModel.inputs[0].name;
  let out = await ingModel.executeAsync({ [inputName]: inputTensor });
  let y = Array.isArray(out) ? out[0] : out; // normalmente [1, C, N] o [1, N, C]

  // Si la forma es [1, C, N], la transponemos a [1, N, C]
  if (y.shape.length === 3 && y.shape[1] < y.shape[2]) {
    y = y.transpose([0, 2, 1]);
  }

  const [batch, num, depth] = y.shape; // [1, N, 4+1+numClases]
  const data = await y.data();

  const boxes = [];

  // Escala de 640x640 (input del modelo) al tamaño real del canvas
  const scaleX = origWidth  / ING_IMG_SIZE;
  const scaleY = origHeight / ING_IMG_SIZE;

  for (let i = 0; i < num; i++) {
    const base = i * depth;

    const cxLogit  = data[base + 0];
    const cyLogit  = data[base + 1];
    const wLogit   = data[base + 2];
    const hLogit   = data[base + 3];
    const objLogit = data[base + 4];

    // Confianza del objeto
    const objConf = sigmoid(objLogit);

    // 1) filtro suave de objeto
    if (objConf < YOLO_OBJ_THRESHOLD) continue;

    // Buscar mejor clase para esta caja
    let bestClass = -1;
    let bestScore = 0;

    for (let c = 5; c < depth; c++) {
      const clsLogit = data[base + c];
      const clsProb  = sigmoid(clsLogit);    // 0–1
      const score    = objConf * clsProb;    // 0–1

      if (score > bestScore) {
        bestScore = score;
        bestClass = c - 5;
      }
    }

    // 2) filtro de score final (obj * clase)
    if (bestClass < 0 || bestScore < YOLO_SCORE_THRESHOLD) continue;

    // Coordenadas en el espacio 640x640 del modelo
    const cx = cxLogit;
    const cy = cyLogit;
    const w  = Math.abs(wLogit);
    const h  = Math.abs(hLogit);

    // Reescalar a tamaño real del canvas
    const xCenter = cx * scaleX;
    const yCenter = cy * scaleY;
    const bw = w * scaleX;
    const bh = h * scaleY;

    const x1 = xCenter - bw / 2;
    const y1 = yCenter - bh / 2;
    const x2 = xCenter + bw / 2;
    const y2 = yCenter + bh / 2;

    boxes.push({
      x1, y1, x2, y2,
      score: bestScore,
      classId: bestClass
    });
  }

  // NMS y limitar número de cajas mostradas
  const finalBoxes = nonMaxSuppression(boxes);
  finalBoxes.sort((a, b) => b.score - a.score);
  const shownBoxes = finalBoxes.slice(0, YOLO_MAX_BOXES);

  // Redibujar la imagen base antes de pintar cajas
  snapshotCtx.drawImage(videoElement, 0, 0, origWidth, origHeight);

  snapshotCtx.lineWidth = 2;
  snapshotCtx.font = "13px system-ui";
  snapshotCtx.textBaseline = "top";

  // Dibujar cada caja seleccionada
  shownBoxes.forEach(b => {
    const label = ingredientClasses[b.classId] ?? `cls ${b.classId}`;
    const pct   = (b.score * 100).toFixed(1);

    const boxW = b.x2 - b.x1;
    const boxH = b.y2 - b.y1;

    // rectángulo
    snapshotCtx.strokeStyle = "#22c55e";
    snapshotCtx.strokeRect(b.x1, b.y1, boxW, boxH);

    // fondo del texto
    const text  = `${label} ${pct}%`;
    const pad   = 2;
    const m     = snapshotCtx.measureText(text);
    const textW = m.width + pad * 2;
    const textH = 16;

    snapshotCtx.fillStyle = "rgba(15,23,42,0.85)";
    snapshotCtx.fillRect(b.x1, b.y1 - textH, textW, textH);

    // texto
    snapshotCtx.fillStyle = "#e5e7eb";
    snapshotCtx.fillText(text, b.x1 + pad, b.y1 - textH + 2);
  });

  // Actualizar lista textual debajo del canvas
  const listDiv = document.getElementById("ingredientsList");
  if (!shownBoxes.length) {
    listDiv.classList.add("muted");
    listDiv.textContent = "No se detectaron ingredientes con confianza suficiente.";
  } else {
    listDiv.classList.remove("muted");
    listDiv.innerHTML = shownBoxes
      .map(b => {
        const name = ingredientClasses[b.classId] ?? `cls ${b.classId}`;
        const pct  = (b.score * 100).toFixed(1);
        return `<div class="pred-item"><b>${name}</b> — ${pct}%</div>`;
      })
      .join("");
  }

  if (Array.isArray(out)) out.forEach(t => t.dispose()); else out.dispose();
}


// ----------------------------
// Predicción conjunta
// ----------------------------

async function predictFromCamera() {
  if (!plateModel) {
    alert("El modelo de platos aún no se ha cargado.");
    return;
  }
  if (!videoElement) return;

  const size = captureFrameToCanvas();
  if (!size) {
    alert("No se pudo capturar el frame de la cámara.");
    return;
  }

  const { width: w, height: h } = size;

  // Usamos un scope para liberar tensores automáticamente
  await tf.nextFrame();
  await tf.engine().startScope();

  const baseImg = tf.browser.fromPixels(snapshotCanvas).toFloat();
  const plateInput = makeTensorForPlate(baseImg);
  const yoloInput  = ingModel ? makeTensorForYolo(baseImg) : null;

  // Clasificación de plato
  await runPlatePrediction(plateInput);

  // Ingredientes
  if (ingModel && yoloInput) {
    await runIngredientsDetection(yoloInput, w, h);
  } else {
    const listDiv = document.getElementById("ingredientsList");
    listDiv.classList.add("muted");
    listDiv.textContent = "Modelo de ingredientes no cargado.";
  }

  baseImg.dispose();
  tf.engine().endScope();
}

// ----------------------------
// Init UI
// ----------------------------

window.addEventListener("DOMContentLoaded", async () => {
  snapshotCanvas = document.getElementById("snapshotCanvas");
  snapshotCtx = snapshotCanvas.getContext("2d");

  try {
    await Promise.all([
      loadPlateModel(),
      loadIngModel(),
      loadPlateClasses(),
      loadIngredientClasses()
    ]);
  } catch (e) {
    console.error("Error cargando modelos o clases:", e);
    const plateDiv = document.getElementById("plateResult");
    plateDiv.classList.add("error-text");
    plateDiv.textContent = "Error cargando modelos. Revisa la consola.";
  }

  await initCamera();

  document.getElementById("predictBtn").addEventListener("click", predictFromCamera);

  const switchBtn = document.getElementById("switchBtn");
  if (switchBtn) switchBtn.addEventListener("click", switchCamera);

  const toggleTop = document.getElementById("toggleTop");
  const topLabel  = document.getElementById("topLabel");
  if (toggleTop) {
    toggleTop.addEventListener("change", (e) => {
      showTop5 = e.target.checked;
      if (topLabel) topLabel.textContent = showTop5 ? "Top-5 platos" : "Top-1 plato";
    });
  }
});
