"""
Validation script — tests the FastAPI /health and /predict endpoints.
Run from the project root: python backend/test_api.py
"""
import json
import os
import urllib.request

BOUNDARY = "----WebKitFormBoundary7MA4YWxkTrZu0gW"


def predict(path: str) -> dict:
    filename = os.path.basename(path)
    with open(path, "rb") as f:
        file_data = f.read()

    body = (
        ("--" + BOUNDARY + "\r\n").encode()
        + ('Content-Disposition: form-data; name="file"; filename="' + filename + '"\r\n').encode()
        + b"Content-Type: image/jpeg\r\n\r\n"
        + file_data
        + ("\r\n--" + BOUNDARY + "--\r\n").encode()
    )

    req = urllib.request.Request(
        "http://localhost:8000/predict",
        data=body,
        headers={"Content-Type": "multipart/form-data; boundary=" + BOUNDARY},
        method="POST",
    )
    r = urllib.request.urlopen(req)
    return json.loads(r.read())


def test_health():
    r = urllib.request.urlopen("http://localhost:8000/health")
    data = json.loads(r.read())
    print("GET /health:", json.dumps(data, indent=2))
    print()


def test_predict():
    tests = [
        ("Dataset/Healthy/1.jpeg",       "Healthy",   0.9801),
        ("Dataset/Potassium/4 (1).jpeg", "Magnesium", 0.7408),
        ("Dataset/Nitrogen/2 (1).jpeg",  "Nitrogen",  0.9983),
    ]

    for path, expected_class, expected_conf in tests:
        result = predict(path)
        pc = result["predicted_class"]
        conf = result["confidence"]
        probs = result["probabilities"]
        match = "[PASS]" if pc == expected_class else "[FAIL]"
        print(f"{match}  [{path}]")
        print(f"   predicted_class : {pc}  (expected: {expected_class})")
        print(f"   confidence      : {conf:.4f}  ({conf * 100:.2f}%)")
        print("   probabilities:")
        for cls, p in probs.items():
            print(f"     {cls:<12}: {p:.4f} ({p * 100:.2f}%)")
        print()


if __name__ == "__main__":
    test_health()
    test_predict()
