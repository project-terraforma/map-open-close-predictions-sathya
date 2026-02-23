from run_model import load_model, run_inference, classify_status
import cv2

# Load the model
model = load_model()

# Run on the downloaded sample
print("Running inference on sample storefront...")
detections = run_inference("test_storefront.jpg", model)

# Print raw generic detections (COCO classes: person, car, etc.)
print(f"Raw Detections (Top 5): {detections[:5]}")

# Run our classifier logic
# Note: Since this is a generic model, it won't find 'For Lease' tags yet.
# outputting the Mock Status to show how the pipeline WOULD react.
status, conf = classify_status(detections, mock_mode=True)
print(f"\nPipeline Classification: {status} (Confidence: {conf})")
