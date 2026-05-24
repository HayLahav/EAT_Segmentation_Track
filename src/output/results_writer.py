"""
Results writer — saves per-cell features and benchmark metrics to CSV / JSON.
No database required. All outputs go to the results/ directory.
"""
import json
import pandas as pd


def write_cell_csv(records: list, output_path: str) -> None:
    """
    Write per-cell feature records to CSV.
    One row per detected cell per frame.

    Columns written (all features present in records):
        cell_id, frame
        Area, Sphericity, x_Pos, y_Pos               (morphology)
        Ruffle_Density, Ruffle_Directionality,
        Protrusion_Length                             (ruffle features)
        Bleb_Count, Bleb_Area_Fraction,
        Bleb_Mean_Circularity, Bleb_Formation_Rate,
        Bleb_Lifetime_Mean, Bleb_Perimeter_Fraction   (bleb features)
        Boundary_Uncertainty_Score                    (FSS output)
    """
    pd.DataFrame(records).to_csv(output_path, index=False)
    print(f"  Saved cell records → {output_path}  ({len(records)} rows)")


def write_metrics_json(metrics: dict, output_path: str) -> None:
    """Write benchmark evaluation metrics (IoU, F1, BF1) to JSON."""
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics     → {output_path}")
