"""
SatQuery AI -- Contract smoke-test for Person 5's functions in models_registry.py.

Validates that run_vqa and run_grounding return EXACTLY the keys the Agentic
Controller (Pair 2) expects, with the right shapes and value types.

Usage:
    .venv\\Scripts\\python.exe test_contract.py [image_path] [download_models]

Run with model download (first time) or offline (uses cached models).
"""
import sys
import os
import json


def validate_vqa(result):
    assert isinstance(result, dict), "run_vqa must return a dict"
    assert set(result.keys()) == {"text_response", "confidence"}, (
        f"run_vqa keys mismatch: {sorted(result.keys())}"
    )
    assert isinstance(result["text_response"], str)
    assert isinstance(result["confidence"], (int, float))
    assert result["confidence"] >= 0.0
    print("  [OK] run_vqa contract valid")
    return True


def validate_grounding(result):
    assert isinstance(result, dict), "run_grounding must return a dict"
    assert set(result.keys()) == {
        "text_response",
        "bounding_boxes",
        "confidence",
    }, f"run_grounding keys mismatch: {sorted(result.keys())}"
    assert isinstance(result["text_response"], str)
    assert isinstance(result["bounding_boxes"], list)
    for b in result["bounding_boxes"]:
        assert set(b.keys()) == {"label", "box_2d", "confidence"}, (
            f"box keys mismatch: {sorted(b.keys())}"
        )
        box2d = b["box_2d"]
        assert isinstance(box2d, (list, tuple)) and len(box2d) == 4, "box_2d must be [ymin, xmin, ymax, xmax]"
        assert any(isinstance(v, (int, float)) for v in box2d)
        assert 0.0 <= b["confidence"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0
    print("  [OK] run_grounding contract valid")
    return True


if __name__ == "__main__":
    from models_registry import run_vqa, run_grounding, load_image

    image = sys.argv[1] if len(sys.argv) > 1 else "RS/before.png"
    print(f"Target image: {image}")
    img = load_image(image)
    print(f"  loaded image -> size {img.size} mode {img.mode}")

    print("Testing run_vqa('Describe this satellite image') ...")
    vqa_res = run_vqa(image, "Describe this satellite image")
    print("  VQA:", json.dumps(vqa_res, ensure_ascii=False)[:300])
    validate_vqa(vqa_res)

    print("Testing run_grounding('airplane') ...")
    gr_res = run_grounding(image, "airplane")
    print("  Grounding:", json.dumps(gr_res, ensure_ascii=False)[:300])
    validate_grounding(gr_res)

    print("\nALL CONTRACT TESTS PASSED.")
