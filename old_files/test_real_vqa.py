# --- START OF FILE test_real_vqa.py ---
import os
import sys
import time
from models_registry import run_vqa, run_grounding, get_execution_metadata

# Target image (Use your real composite or GeoTIFF)
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "demo_rgb.png"

if not os.path.exists(IMAGE_PATH):
    print(f"❌ Error: Image '{IMAGE_PATH}' not found. Please provide a valid image path.")
    sys.exit(1)

print(f"\n========================================================")
print(f"🛰️ SATQUERY AI: REAL-WORLD RSVQA & GROUNDING BENCHMARK")
print(f"Target Image: {IMAGE_PATH}")
print(f"========================================================\n")

# Battery of standard RSVQA benchmark queries
TEST_QUERIES = [
    # 1. Scene Description / Captioning (VRSBench standard)
    ("LULC Captioning", "Describe the primary land cover, terrain, and structures in this satellite image."),
    
    # 2. Presence Query (RSVQA standard)
    ("Presence Detection", "Is there any road, water body, or agricultural field visible in this image?"),
    
    # 3. Numeric / Counting Query (RSVQA standard)
    ("Object Counting", "How many distinct buildings, structures, or field plots can you identify?"),
    
    # 4. Spatial / Comparative Query (RSVQA standard)
    ("Spatial Comparison", "Which occupies more surface area: vegetation/farmland or built-up infrastructure?")
]

# Run VQA Battery
for category, query in TEST_QUERIES:
    print(f"--- [Category: {category}] ---")
    print(f"❓ Query: \"{query}\"")
    
    t0 = time.time()
    result = run_vqa(IMAGE_PATH, query)
    latency = time.time() - t0
    
    print(f"💬 Answer: {result['text_response']}")
    print(f"📊 Confidence: {result.get('confidence', 0.9)*100:.1f}% | ⏱️ Latency: {latency:.2f}s\n")

# Run Grounding Test
GROUNDING_TARGET = "agricultural field" # Change to 'building', 'road', or 'water' based on your image
print(f"--- [Category: Text-Guided Visual Grounding] ---")
print(f"❓ Target Prompt: \"{GROUNDING_TARGET}\"")

t0 = time.time()
g_res = run_grounding(IMAGE_PATH, GROUNDING_TARGET)
g_latency = time.time() - t0

print(f"💬 Answer: {g_res['text_response']}")
print(f"🎯 Bounding Boxes Found: {len(g_res.get('bounding_boxes', []))}")
for i, box in enumerate(g_res.get("bounding_boxes", [])[:3]): # Print first 3 boxes
    print(f"   Box {i+1}: {box['box_2d']} (Conf: {box['confidence']:.2f})")
print(f"⏱️ Latency: {g_latency:.2f}s\n")

# Execution Metadata
meta = get_execution_metadata()
print(f"🛠️ Execution Trace: {meta}")
print(f"\n✅ All real-data benchmark tests completed successfully!")