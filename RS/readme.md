Developed the Pipeline (BT_CM.py) which serves as the "Scientific Expert" for the SatQuery Assistant. While the main UI handles the conversation, the engine performs the low-level signal processing and structural analysis required to provide "Evidence-Grounded" answers.

TOOLS:
Bi-Temporal Change: Implements a Siamese-style analysis pipeline. It uses sub-pixel image registration (ORB-features) and radiometric normalization (Histogram Matching) to align imagery from two different dates and extract a precise Change Evidence Mask.

Cross-Modal (SAR-Optical): A cloud-penetration logic module. It simulates RISAT-1A (Radar) backscatter signatures to detect water bodies and metallic structures (dams/ships), providing a "Structural Truth" layer that visual optical sensors miss.

Explainable AI (XAI) Dashboard: Generates Saliency Heatmaps (Grad-CAM style) to visualize the AI's neural attention. This proves the system is attending to actual geographic features, ensuring "Auditable Reasoning."

Core Signal Processing 
Lee-Filter Implementation: Applied adaptive noise reduction to handle speckle and sensor grain, significantly improving the Signal-to-Noise Ratio (SNR) for detection.
SAR-Guided Synthesis: Developed a "Cloud-Free View" generator that uses SAR structural edges to "inpain" or reconstruct historical optical data behind monsoonal cloud cover.
Trust Metrics: Every analysis returns a Confidence Score based on the statistical reliability of the image alignment and signal variance.

Auditable Output 
Generates a structured Execution Summary (JSON) after every query like:
change_intensity: The exact % of geographic shift (e.g., "47.48%").
confidence_score: The trust level of the registration (e.g., "1.0").
verdict: The physical interpretation (e.g., "Major Reservoir Inundation").
lat_lon: Geographic grounding for regional identification.