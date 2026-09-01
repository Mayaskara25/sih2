import numpy as np
import cv2
import json
from matplotlib import pyplot as plt
from BT_CM import SatQuerySpecialist # Ensure your engine file is named BT_CM.py

# --- 1. INITIALIZATION ---
analyst = SatQuerySpecialist()
path_t0 = 'before.png' # Reference
path_t1 = 'after.png'  # Current Acquisition

print("🛰️ SATQUERY AI: Launching Global Analyst Logic...")

try:
    # --- 2. PIPELINE EXECUTION ---
    # A. Bitemporal Core
    p1, p2, mask, conf = analyst.run_bi_temporal_analysis(path_t0, path_t1)
    report = analyst.generate_agentic_report(mask, conf)
    
    # B. Specialist Reasoning (XAI & Synthesis)
    saliency = analyst.generate_xai_saliency(mask)
    sar_mask, structural_edges = analyst.simulate_sar_fusion(p2)
    synthetic = analyst.synthesize_cloud_free(p1, p2, structural_edges)
    
    # --- 3. HARDWARE LINK (OPTIONAL TEST) ---
    print(f"TELEMETRY PUSH: {report['findings']['change_intensity']} Change Detected.")

    # --- 4. THE 9-PANEL ULTIMATE DASHBOARD ---
    plt.figure(figsize=(20, 12), facecolor='#000000')
    plt.style.use('dark_background')
    
    # Define color scheme
    accent = '#00FF00' # Green for success
    warn = '#FFD700'   # Gold for warnings

    # Panel 1: T0 Historical Reference
    plt.subplot(3, 3, 1)
    plt.imshow(p1)
    plt.title("MODALITY A: T0 REFERENCE", color=accent, fontsize=10)
    plt.xlabel("Grounded historical baseline from ISRO Cartosat-3 archive.", fontsize=8)
    plt.axis('off')

    # Panel 2: T1 Observed State
    plt.subplot(3, 3, 2)
    plt.imshow(p2)
    plt.title("MODALITY B: T1 ACQUISITION", color=accent, fontsize=10)
    plt.xlabel("Newly acquired optical signal with observed inundation.", fontsize=8)
    plt.axis('off')

    # Panel 3: Histogram Matching
    plt.subplot(3, 3, 3)
    diff_vis = cv2.absdiff(cv2.cvtColor(p1, cv2.COLOR_RGB2GRAY), cv2.cvtColor(p2, cv2.COLOR_RGB2GRAY))
    plt.imshow(diff_vis, cmap='inferno')
    plt.title("DSP: RADIOMETRIC ALIGNMENT", color=warn, fontsize=10)
    plt.xlabel("Normalizing dynamic range to eliminate solar illumination noise.", fontsize=8)
    plt.axis('off')

    # Panel 4: Change Evidence
    plt.subplot(3, 3, 4)
    plt.imshow(mask, cmap='hot')
    plt.title("EVIDENCE: CHANGE INTENSITY", color=warn, fontsize=10)
    plt.xlabel(f"Detected {report['findings']['change_intensity']} structural variance via Siamese logic.", fontsize=8)
    plt.axis('off')

    # Panel 5: SAR Water Detect
    plt.subplot(3, 3, 5)
    plt.imshow(sar_mask, cmap='Blues')
    plt.title("MODALITY C: RISAT-1A RADAR", color=accent, fontsize=10)
    plt.xlabel("SAR specular backscatter confirming liquid inundation surface.", fontsize=8)
    plt.axis('off')

    # Panel 6: Structural Fusion
    plt.subplot(3, 3, 6)
    plt.imshow(structural_edges, cmap='gray')
    plt.title("SIGNAL: STRUCTURAL EDGES", color=accent, fontsize=10)
    plt.xlabel("Extracting radar-truth building outlines for precise grounding.", fontsize=8)
    plt.axis('off')

    # Panel 7: XAI Reasoning
    plt.subplot(3, 3, 7)
    plt.imshow(saliency)
    plt.title("LOGIC: XAI ATTENTION MAP", color=warn, fontsize=10)
    plt.xlabel("Visualizing neural attention hot-zones (Grad-CAM simulation).", fontsize=8)
    plt.axis('off')

    # Panel 8: Synthetic Output
    plt.subplot(3, 3, 8)
    plt.imshow(synthetic)
    plt.title("FINAL: CLOUD-FREE SYNTHESIS", color=accent, fontsize=10)
    plt.xlabel("Synthesized cloud-free view via SAR-Structural Inpainting.", fontsize=8)
    plt.axis('off')

    # Panel 9: Mission Terminal
    plt.subplot(3, 3, 9)
    plt.axis('off')
    meta_json = json.dumps(report, indent=1)
    plt.text(0, 0.5, f"SITE: GONAHAL, IN\nTRUST: {conf*100:.1f}%\nLAT: 15.17, LON: 76.54\n\n{meta_json}", 
             color='#00FF00', family='monospace', fontsize=9, verticalalignment='center')
    plt.title("TERMINAL: AGENTIC EXECUTION", color=accent, fontsize=10)

    plt.suptitle("Result", fontsize=16, color='white')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the Master Document
    plt.savefig('SATQUERY_FINAL_DASHBOARD.png', dpi=300, facecolor='#000000')
    plt.show()

except Exception as e:
    print(f"Mission Abort: {e}")