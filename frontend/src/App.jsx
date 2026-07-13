import { useState } from "react";
import "./App.css";
import ImageUploader from "./components/ImageUploader";
import PredictionResult from "./components/PredictionResult";

const API_URL = "http://localhost:8000/predict";

function App() {
  // Holds the image file selected by the user, plus a local preview URL
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  // Holds the parsed response from the backend once a prediction is made
  const [prediction, setPrediction] = useState(null);

  // UI state for the request lifecycle
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageSelect = (file) => {
    setSelectedImage(file);
    setPreviewUrl(URL.createObjectURL(file));
    setPrediction(null);
    setError(null);
  };

  const handlePredict = async () => {
    if (!selectedImage) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedImage);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Backend returned status ${response.status}`);
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError("Could not reach the prediction server. Is the backend running?");
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>CIFAR-10 Image Classifier</h1>
        <p>Upload an image to see what the model predicts</p>
      </header>

      <div className="card">
        <ImageUploader
          previewUrl={previewUrl}
          onImageSelect={handleImageSelect}
          onPredict={handlePredict}
          isLoading={isLoading}
          hasImage={!!selectedImage}
        />

        {error && <p className="error-message">{error}</p>}

        {prediction && <PredictionResult prediction={prediction} />}
      </div>
    </div>
  );
}

export default App;