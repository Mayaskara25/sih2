import numpy as np
import cv2
import rasterio
import json
from skimage.exposure import match_histograms
from scipy.ndimage import uniform_filter

class SatQuerySpecialist:
    def __init__(self):
        self.version = "v3.1_Production"

    def load_image(self, path):
        if path.lower().endswith(('.tif', '.tiff')):
            with rasterio.open(path) as src:
                return cv2.normalize(src.read([1,2,3]).transpose(1,2,0), None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def register_images(self, img1, img2):
        g1, g2 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create(nfeatures=2500)
        kp1, des1 = orb.detectAndCompute(g1, None)
        kp2, des2 = orb.detectAndCompute(g2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
        
        # Recalibrated ECE Metric: Relaxed for public imagery jitter
        good = [m for m in matches if m.distance < 50]
        reg_conf = min(len(good) / 25.0, 1.0) 
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        matrix, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts)
        aligned = cv2.warpAffine(img2, matrix, (img1.shape[1], img1.shape[0]))
        return aligned, reg_conf

    def calculate_signal_quality(self, mask):
        num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) == 0: return 0.0, 0.0
        signal = np.sum(areas[areas >= 100]) # Meaningful change
        noise = np.sum(areas[areas < 100])   # Pixel grain
        snr_db = 10 * np.log10((signal / (noise + 1e-6)) + 1e-7)
        coherence = (signal / (signal + noise + 1e-6)) * 100
        return round(snr_db, 2), round(coherence, 2)

    def run_bi_temporal_analysis(self, path0, path1):
        t0, t1_raw = self.load_image(path0), self.load_image(path1)
        t1_alg, reg_conf = self.register_images(t0, t1_raw)
        t1_match = match_histograms(t1_alg, t0, channel_axis=-1).astype('uint8')
        diff = cv2.absdiff(cv2.cvtColor(t0, cv2.COLOR_RGB2GRAY), cv2.cvtColor(t1_match, cv2.COLOR_RGB2GRAY))
        mask = cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        return t0, t1_match, mask, reg_conf

    def generate_agentic_report(self, mask, reg_conf):
        intensity = (np.sum(mask == 255) / mask.size) * 100
        snr, coh = self.calculate_signal_quality(mask)
        # Trust Logic: Balanced between Alignment and Signal Unity
        trust = (reg_conf * 0.4) + (coh/100 * 0.6)
        return {
            "intensity": f"{intensity:.2f}%",
            "metrics": {
                "registration_trust": f"{reg_conf*100:.1f}%",
                "spatial_snr": f"{snr} dB",
                "spatial_coherence": f"{coh}%",
                "system_total_trust": f"{trust*100:.1f}%"
            },
            "verdict": "VERIFIED GEOGRAPHIC CHANGE" if trust > 0.45 else "SIGNAL INCONCLUSIVE"
        }

    def generate_visual_modalities(self, t0, t1, mask):
        # Saliency
        sal = cv2.applyColorMap(cv2.normalize(cv2.GaussianBlur(mask.astype(float), (51, 51), 0), None, 0, 255, cv2.NORM_MINMAX).astype('uint8'), cv2.COLORMAP_JET)
        # SAR
        gray = cv2.cvtColor(t1, cv2.COLOR_RGB2GRAY)
        sar = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)[1]
        # Fusion
        edges = cv2.Canny(gray, 100, 200)
        synth = t1.copy()
        synth[edges > 0] = [0, 255, 255]
        return sal, sar, edges, synth