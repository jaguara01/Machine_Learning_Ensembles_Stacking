import torch

# Path to your processed file
PT_FILE = "data/processed_fonts.pt"


def inspect_data():
    print(f"📂 Loading {PT_FILE}...")

    # 1. Load the dictionary
    data = torch.load(PT_FILE)

    # 2. Extract keys
    class_names = data["class_names"]
    labels = data["y"]

    print("\n📝 List of Fonts (Classes):")
    print("-" * 30)

    # 3. Print each font with its index
    for idx, name in enumerate(class_names):
        print(f"  [{idx}] {name}")

    print("-" * 30)
    print(f"✅ Total Classes: {len(class_names)}")

    # Optional: Print how many images you have per font
    print("\n📊 Sample Counts per Font:")
    # Count occurrences of each label index
    unique_labels, counts = torch.unique(labels, return_counts=True)

    for label_idx, count in zip(unique_labels, counts):
        font_name = class_names[label_idx]
        print(f"  - {font_name}: {count.item()} images")

    print("\n📊 Total Sample Counts:")
    print(f"Total from counts: {counts.sum().item()}")


if __name__ == "__main__":
    inspect_data()
