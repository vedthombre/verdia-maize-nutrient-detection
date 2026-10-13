from datasets import load_dataset

ds = load_dataset("surfiniaburger/maize-leaf-health-dataset")

print(ds)
print("Splits:", list(ds.keys()))

for split, data in ds.items():
    print(f"\nSplit: {split}")
    print("Columns:", data.column_names)
    print("Features:", data.features)

    label_column = "label" if "label" in data.column_names else "text"
    print("Label counts:", data.to_pandas()[label_column].value_counts())