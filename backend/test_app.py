"""
backend/test_app.py
===================
Validates backend/app.py endpoints against the verified predict.py reference values.

Run from the project root:
    python backend/test_app.py
"""

import io
import json
import os
import urllib.request

BASE = "http://localhost:8000"
BOUNDARY = "----VerdiaBoundary7MA4YWxkTrZu0gW"

# Reference values from verified predict.py runs
TESTS = [
    {
        "path":             "Dataset/Healthy/1.jpeg",
        "expected_class":   "Healthy",
        "expected_conf_pct": 98.01,
    },
    {
        "path":             "Dataset/Nitrogen/2 (1).jpeg",
        "expected_class":   "Nitrogen",
        "expected_conf_pct": 99.83,
    },
    {
        "path":             "Dataset/Potassium/4 (1).jpeg",
        "expected_class":   "Magnesium",
        "expected_conf_pct": 74.08,
    },
]


def _multipart_post(url: str, file_path: str) -> dict:
    """Send a multipart/form-data POST with a single 'file' field."""
    filename = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        file_data = f.read()

    body = (
        ("--" + BOUNDARY + "\r\n").encode()
        + ('Content-Disposition: form-data; name="file"; filename="' + filename + '"\r\n').encode()
        + b"Content-Type: image/jpeg\r\n\r\n"
        + file_data
        + ("\r\n--" + BOUNDARY + "--\r\n").encode()
    )

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "multipart/form-data; boundary=" + BOUNDARY},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_health():
    print("=" * 60)
    print("GET /health")
    print("=" * 60)
    with urllib.request.urlopen(BASE + "/health") as r:
        data = json.loads(r.read())
    print(json.dumps(data, indent=2))
    assert data["status"] == "ok", "Health check failed!"
    print("[PASS] /health\n")


def test_predict():
    print("=" * 60)
    print("POST /predict — three validation images")
    print("=" * 60)

    all_pass = True
    for t in TESTS:
        result = _multipart_post(BASE + "/predict", t["path"])

        pred    = result["prediction"]
        conf    = result["confidence"]
        probs   = result["probabilities"]
        conf_pct = round(conf * 100, 2)

        ok = (pred == t["expected_class"])
        tag = "[PASS]" if ok else "[FAIL]"
        if not ok:
            all_pass = False

        print(f"{tag}  {t['path']}")
        print(f"   prediction   : {pred}  (expected: {t['expected_class']})")
        print(f"   confidence   : {conf:.4f}  ({conf_pct:.2f}%)")
        print("   probabilities:")
        for cls, p in probs.items():
            print(f"     {cls:<12}: {p:.4f} ({p * 100:.2f}%)")
        print()

    print("All tests PASSED." if all_pass else "SOME TESTS FAILED.")


if __name__ == "__main__":
    test_health()
    test_predict()
