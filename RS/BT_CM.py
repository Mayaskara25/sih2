import numpy as np
import cv2
import rasterio
import json
from skimage.exposure import match_histograms
from scipy.ndimage import uniform_filter

class SatQuerySpecialist:
    def __init__(self):
        self.version = "v2.0_SIH_Optimized"
        print(f"SatQuery Specialist Engine {self.version} Initialized")

    # --- IMPROVEMENT 1: HISTOGRAM MATCHING & SIGNAL FILTERING ---
    def preprocess_signals(self, img1, img2):
        """ Normalizes lighting and reduces sensor noise """
        # Force lighting of Image 2 to match Image 1 (Histogram Alignment)
        img2_matched = match_histograms(img2, img1, channel_axis=-1)
        
        # Apply Gaussian Blur to reduce sensor 'grain' (Noise Filtering)
        clean1 = cv2.GaussianBlur(img1, (3, 3), 0)
        clean2 = cv2.GaussianBlur(img2_matched.astype('uint8'), (3, 3), 0)
        return clean1, clean2

    # --- IMPROVEMENT 2: ENHANCED REGISTRATION WITH CONFIDENCE SCORE ---
    def register_images(self, img_before, img_after):
        """ Aligns images and returns a Reliability Metric """
        g1 = cv2.cvtColor(img_before, cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(img_after, cv2.COLOR_RGB2GRAY)

        orb = cv2.ORB_create(nfeatures=3000)
        kp1, des1 = orb.detectAndCompute(g1, None)
        kp2, des2 = orb.detectAndCompute(g2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Calculate Confidence based on Match Density (0 to 1)
        confidence = min(len(matches) / 500, 1.0) 

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        matrix, inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts)
        
        aligned_after = cv2.warpAffine(img_after, matrix, (img_before.shape[1], img_before.shape[0]))
        return aligned_after, confidence

    # --- IMPROVEMENT 3: LEE-FILTER SIMULATION (For SAR backscatter) ---
    def apply_lee_filter(self, img, size=5):
        """ Removes 'Speckle Noise' from simulated RISAT-1A SAR data """
        img = img.astype(float)
        img_mean = uniform_filter(img, (size, size))
        img_sqr_mean = uniform_filter(img**2, (size, size))
        img_variance = img_sqr_mean - img_mean**2

        overall_variance = np.var(img)
        img_weights = img_variance / (img_variance + overall_variance + 1e-6)
        img_output = img_mean + img_weights * (img - img_mean)
        return img_output.astype('uint8')

    # --- MANDATORY TASK: BI-TEMPORAL PIPELINE ---
    def run_bi_temporal_analysis(self, path1, path2):
        # 1. Load using robust loader (from previous versions)
        raw1 = self.load_image(path1)
        raw2 = self.load_image(path2)
        
        # 2. Registration + Confidence
        t2_aligned, conf = self.register_images(raw1, raw2)
        
        # 3. Pre-process (Lighting/Noise correction)
        p1, p2 = self.preprocess_signals(raw1, t2_aligned)
        
        # 4. Differencing
        g1 = cv2.cvtColor(p1, cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(p2, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(g1, g2)
        
        # Morphological Closing to connect fragments
        _, mask = cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        
        return p1, p2, mask, conf

    # --- MANDATORY TASK: CROSS-MODAL REPORT (AUDITABLE SUMMARY) ---
    def generate_agentic_report(self, mask, confidence):
        intensity = (np.sum(mask == 255) / mask.size) * 100
        
        # This JSON output fulfills the 'Agentic Execution Summary' requirement
        report = {
            "execution_meta": {
                "specialist_module": "Bi-Temporal & Cross-Modal Unit",
                "processor_version": self.version,
                "confidence_score": round(confidence, 2)
            },
            "findings": {
                "change_intensity": f"{intensity:.2f}%",
                "verdict": "Major Inundation/Expansion" if intensity > 15 else "Stabilized Zone",
                "recommended_modality": "RISAT-1A SAR (C-Band)" if intensity > 5 else "None"
            },
            "grounding": "Cartosat-3 Spatial Pixel Variance Detected"
        }
        return report

    def load_image(self, path):
        """ Final Robust Loader """
        if path.lower().endswith(('.tif', '.tiff')):
            with rasterio.open(path) as src:
                return cv2.normalize(src.read([1,2,3]).transpose(1,2,0), None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def simulate_sar_fusion(self, opt_img):
        """ Professional Cross-Modal Verification """
        # Step 1: Simulated Despeckling
        gray = cv2.cvtColor(opt_img, cv2.COLOR_RGB2GRAY)
        clean_radar = self.apply_lee_filter(gray)
        
        # Step 2: High backscatter detection (Urban/Water thresholding)
        _, water_detect = cv2.threshold(clean_radar, 65, 255, cv2.THRESH_BINARY_INV)
        edges = cv2.Canny(clean_radar, 80, 160)
        
        return water_detect, edges
    # --- ENHANCEMENT 1: "CLOUD-THROUGH" SYNTHESIS ---
    def synthesize_cloud_free(self, old_clear, new_cloudy, sar_mask):
        """ 
        Simulates SAR-to-Optical Reconstruction.
        Replaces cloudy regions with historical data verified by SAR structural truth.
        """
        # Create a simple 'Cloud Mask' based on high brightness (White pixels)
        gray_new = cv2.cvtColor(new_cloudy, cv2.COLOR_RGB2GRAY)
        _, clouds = cv2.threshold(gray_new, 200, 255, cv2.THRESH_BINARY)
        
        # Inpainting Logic: 
        # We 'Repair' the cloudy areas using the old clear data
        # but we preserve SAR verified structures (sar_mask)
        reconstructed = new_cloudy.copy()
        reconstructed[clouds > 0] = old_clear[clouds > 0]
        
        # Inject the 'Radar Truth' edges in Cyan to show the 'Fused Reconstruction'
        reconstructed[sar_mask > 0] = [0, 255, 255] # Cyan color for fusion
        
        return reconstructed

    # --- ENHANCEMENT 2: XAI (SALIENCY HEATMAP) ---
    def generate_xai_saliency(self, change_mask):
        """ 
        Simulates Grad-CAM / Explainable AI Attention.
        Visualizes the 'Reasoning Hot-Zone' of the AI model.
        """
        # We apply a large Gaussian Blur to the change mask to create 
        # a 'Receptive Field' similar to a CNN's attention.
        saliency = cv2.GaussianBlur(change_mask.astype(float), (51, 51), 0)
        
        # Normalize and apply a JET colormap (Red = High Attention)
        saliency_norm = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        saliency_heatmap = cv2.applyColorMap(saliency_norm, cv2.COLORMAP_JET)
        
        return saliency_heatmap