/**
 * PredictionResult
 *
 * Displays the model's top prediction and a horizontal bar chart
 * showing confidence scores for all 10 CIFAR-10 classes, sorted
 * from highest to lowest confidence.
 *
 * Note: the backend returns confidence values already as percentages
 * (e.g. 95.37, not 0.9537), so no additional multiplication is needed.
 */
function PredictionResult({ prediction }) {
  const { predicted_class, confidence, all_confidences } = prediction;

  // Sort classes by confidence, highest first
  const sortedClasses = Object.entries(all_confidences).sort(
    (a, b) => b[1] - a[1]
  );

  return (
    <div className="result">
      <div className="top-prediction">
        <span className="top-prediction-label">Prediction</span>
        <h2>{predicted_class}</h2>
        <span className="top-prediction-confidence">
          {confidence.toFixed(1)}% confidence
        </span>
      </div>

      <div className="confidence-bars">
        {sortedClasses.map(([className, score]) => (
          <div key={className} className="bar-row">
            <span className="bar-label">{className}</span>
            <div className="bar-track">
              <div
                className={`bar-fill ${
                  className === predicted_class ? "bar-fill-top" : ""
                }`}
                style={{ width: `${score}%` }}
              />
            </div>
            <span className="bar-value">{score.toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default PredictionResult;