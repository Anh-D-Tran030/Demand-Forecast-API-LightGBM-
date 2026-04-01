from pathlib import Path

from sklearn.datasets import fetch_california_housing


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = fetch_california_housing(as_frame=True)
    frame = dataset.frame

    output_file = output_dir / "california_housing.csv"
    frame.to_csv(output_file, index=False)
    print(f"Saved dataset to {output_file}")


if __name__ == "__main__":
    main()
