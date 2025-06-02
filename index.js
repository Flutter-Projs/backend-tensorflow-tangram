const express = require("express");
const app = express();
const tf = require("@tensorflow/tfjs-node");
const bodyParser = require("body-parser");
const fs = require("fs");
const path = require("path");
   const cors = require('cors');
  //  app.use(cors({ origin: 'http://localhost:4200' }));
  //  app.use(cors({ origin: 'http://localhost:38247' }));
   app.use(cors({ origin: 'http://localhost:52009' }));

app.use(bodyParser.json({ limit: "10mb" }));

let model;
const MODEL_PATH = path.join(__dirname, "public/assets/my_model/model.json");

async function loadModel() {
  if (!model) {
    model = await tf.loadLayersModel("file://" + MODEL_PATH);
  }
}

// Endpoint POST para recibir datos de entrada y devolver la predicción (tensor) 
app.post('/predict', async (req, res) => {
  try {
    await loadModel();
    const { inputData, shape } = req.body;
    if (!inputData || !shape) {
      return res.status(400).json({ error: 'No inputData or shape provided' });
    }

    // Reconstruye el tensor desde inputData y shape
    const tensor = tf.tensor(inputData, shape);

    // Realiza la predicción
    const prediction = model.predict(tensor);
    const result = prediction.dataSync();

    res.json({ result: Array.from(result) });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Error processing input tensor' });
  }
});

// Endpoint POST para recibir imagen base64 en el cuerpo de la solicitud
app.post('/predict', async (req, res) => {
  try {
    await loadModel();
    const { imageBase64 } = req.body;
    if (!imageBase64) {
      return res.status(400).json({ error: 'No imageBase64 provided' });
    }

    // Decodifica la imagen base64 a un buffer
    const buffer = Buffer.from(imageBase64, 'base64');
    // Decodifica el buffer a un tensor, redimensiona, convierte a float, normaliza y expande dimensión
    let tensor = tf.node.decodeImage(buffer).resizeNearestNeighbor([224, 224]).toFloat().expandDims(0);
    // Si tiene 4 canales (RGBA), conviértelo a 3 canales (RGB)
    if (tensor.shape[3] === 4) {
      tensor = tensor.slice([0, 0, 0, 0], [-1, -1, -1, 3]);
    }
    tensor = tensor.div(255.0);

    // Realiza la predicción
    const prediction = model.predict(tensor);
    const result = prediction.dataSync();

    res.json({ result: Array.from(result) });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Error processing image' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
