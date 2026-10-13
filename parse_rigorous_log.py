import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("d:/Antigravity Code/DL_Project/gradcam_rigorous_results")
with open(RESULTS_DIR / "rigorous_log.json", "r") as f:
    log_data = json.load(f)

total_images = len(log_data)
print(f"Total images tested: {total_images}")

# Categories
# 1. Original vs Black BG
# 2. Original vs Neutral BG
# 3. Original vs Disrupted Leaf

def analyze_condition(cond_key):
    flips_correct_to_incorrect = 0
    flips_incorrect_to_correct = 0
    unchanged = 0
    class_wise_flips = defaultdict(int)
    
    for item in log_data:
        orig = item["original"]
        cond = item[cond_key]
        true_cls = item["true_class"]
        
        if orig["pred"] == cond["pred"]:
            unchanged += 1
        elif orig["correct"] and not cond["correct"]:
            flips_correct_to_incorrect += 1
            class_wise_flips[true_cls] += 1
        elif not orig["correct"] and cond["correct"]:
            flips_incorrect_to_correct += 1
            class_wise_flips[true_cls] += 1
        else:
            # incorrect to different incorrect
            pass
            
    print(f"\\n--- Analysis for {cond_key} ---")
    print(f"Unchanged predictions: {unchanged}")
    print(f"Correct -> Incorrect flips: {flips_correct_to_incorrect}")
    print(f"Incorrect -> Correct flips: {flips_incorrect_to_correct}")
    print(f"Class-wise flips (any change involving correct): {dict(class_wise_flips)}")

analyze_condition("bg_black")
analyze_condition("bg_neutral")
analyze_condition("leaf_disrupted")
