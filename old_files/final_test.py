import os
import json
import time
import cv2
import matplotlib.pyplot as plt
from BT_CM import SatQuerySpecialist
from models_registry import run_vqa, get_execution_metadata
import numpy as np

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    clear_screen()
    print("="*60)
    print("🛰️  SATQUERY AI: UNIVERSAL AGENTIC INTERFACE")
    print("MODALITY: OPTICAL / SAR / BI-TEMPORAL")
    print("="*60)

    # 1. USER INPUT PHASE (Zero Hardcoding)
    print("\n[INPUT SECTION]")
    path1 = input("📁 Drag & Drop IMAGE 1 (Reference/Current): ").strip().replace("'", "").replace('"', '')
    path2 = input("📁 Drag & Drop IMAGE 2 (Optional - for Change Detection, else press Enter): ").strip().replace("'", "").replace('"', '')
    user_prompt = input("💬 Ask SatQuery a question about this scene: ")

    if not os.path.exists(path1):
        print("❌ Error: Image 1 path is invalid. Exiting.")
        return

    # 2. ANALYST ENGINE INITIALIZATION (PERSON 6)
    analyst = SatQuerySpecialist()
    
    # 3. TASK SELECTION LOGIC (AGENTIC ORCHESTRATION)
    print("\n[PROCESS] Agent selecting specialist tools...")
    
    if path2 and os.path.exists(path2):
        # --- BITEMPORAL WORKFLOW ---
        mode = "BI-TEMPORAL"
        print(f"-> Logic: Executing Siamese Differencing on {os.path.basename(path1)} & {os.path.basename(path2)}")
        t0, t1, mask, reg_conf = analyst.run_bi_temporal_analysis(path1, path2)
        report = analyst.generate_agentic_report(mask, reg_conf)
    else:
        # --- SINGLE IMAGE WORKFLOW ---
        mode = "SINGLE-IMAGE"
        print(f"-> Logic: Executing Scene Grounding on {os.path.basename(path1)}")
        t1 = analyst.load_image(path1)
        t0 = t1.copy() # Placeholder for display
        mask = np.zeros(t1.shape[:2], dtype=np.uint8) # No change mask for single image
        reg_conf = 1.0
        report = {
            "intensity": "N/A (Single Image)",
            "metrics": {"registration_trust": "100%", "spatial_snr": "N/A", "system_total_trust": "100%"},
            "verdict": "Single Scene Contextual Analysis"
        }

    # 4. CROSS-MODAL & XAI FUSION (PERSON 6)
    sal, sar, edg, syn = analyst.generate_visual_modalities(t0, t1, mask)

    # 5. VLM REASONING (PERSON 5)
    print(f"[PROCESS] Prompting VLM with Signal Context: {report['intensity']}...")
    context_prompt = f"Grounded Context: {json.dumps(report)}. Query: {user_prompt}"
    vqa_res = run_vqa(path1, context_prompt)

    # 6. TERMINAL MISSION BRIEFING (THE JSON METADATA)
    clear_screen()
    print("\n" + "🚀" + "="*55)
    print(" SATQUERY MISSION TERMINAL: AUDITABLE EXECUTION LOG ")
    print("="*57)
    print(f"MISSION MODALITY   : {mode}")
    print(f"SIGNAL INTENSITY   : {report.get('intensity', 'N/A')}")
    print(f"SYSTEM TRUST       : {report.get('metrics', {}).get('system_total_trust', '100%')}")
    print(f"VLM REASONING      : {vqa_res['text_response']}")
    print("-" * 57)
    
    # Save JSON to disk for full-pipeline audit
    with open('mission_audit.json', 'w') as f:
        json.dump({"metrics": report, "ai_response": vqa_res}, f, indent=4)

    # 7. DASHBOARD DISPLAY
    plt.figure(figsize=(18, 10), facecolor='black')
    plt.style.use('dark_background')
    
    # Optimized 6-Panel Layout for Universal Data
    vis = [t0, t1, mask, sal, sar, syn]
    titles = ["T0 BASELINE", "T1 CURRENT", "CHANGE MASK", "AI ATTENTION (XAI)", "SAR PROXY", "SYNTHETIC FUSION"]

    for i in range(6):
        plt.subplot(2, 3, i+1)
        if i == 2: plt.imshow(vis[i], cmap='hot')
        elif i == 4: plt.imshow(vis[i], cmap='Blues')
        else: plt.imshow(vis[i])
        plt.title(titles[i], color='#00FF00', fontweight='bold')
        plt.axis('off')

    plt.suptitle(f"SatQuery Universal Ground Control | Modality: {mode}", fontsize=18, color='white')
    
    # Grounded reasoning as subtitle
    plt.figtext(0.5, 0.02, f"REASONING: {vqa_res['text_response'][:300]}...", 
                wrap=True, horizontalalignment='center', color='yellow', fontsize=10, 
                bbox=dict(facecolor='blue', alpha=0.1))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    import numpy as np # Ensure numpy is imported for single-image case
    main()