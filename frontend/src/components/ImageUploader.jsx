/**
 * ImageUploader
 *
 * Lets the user pick an image file, shows a preview of it,
 * and triggers the prediction request via the onPredict callback.
 */
function ImageUploader({ previewUrl, onImageSelect, onPredict, isLoading, hasImage }) {
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      onImageSelect(file);
    }
  };

  return (
    <div className="uploader">
      <label className="upload-label" htmlFor="file-input">
        {previewUrl ? (
          <img src={previewUrl} alt="Selected preview" className="image-preview" />
        ) : (
          <div className="upload-placeholder">
            <span>Click to choose an image</span>
          </div>
        )}
      </label>

      <input
        id="file-input"
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="file-input"
      />

      <button
        onClick={onPredict}
        disabled={!hasImage || isLoading}
        className="predict-button"
      >
        {isLoading ? "Predicting..." : "Predict"}
      </button>
    </div>
  );
}

export default ImageUploader;