#!/usr/bin/env python3
"""
Standalone submission validator.

Tests the final submission against competition requirements.
"""

import sys
import pandas as pd
from pathlib import Path

from core.solution import PackingSolution
from infrastructure.validator import SolutionValidator


def validate_csv_format(filepath: str) -> bool:
    """Validate CSV format."""
    print("\n[1] CSV Format Validation")
    print("-" * 70)
    
    df = pd.read_csv(filepath)
    
    # Check columns
    required_cols = {'id', 'x', 'y', 'deg'}
    if not required_cols.issubset(df.columns):
        print(f"✗ Missing columns: {required_cols - set(df.columns)}")
        return False
    print(f"✓ Columns correct: {list(df.columns)}")
    
    # Check 's' prefix
    for col in ['x', 'y', 'deg']:
        if not df[col].astype(str).str.startswith('s').all():
            print(f"✗ Not all {col} values have 's' prefix")
            return False
    print("✓ All numeric values have 's' prefix")
    
    # Check row count (20100 = 1+2+3+...+200)
    expected_rows = sum(range(1, 201))
    if len(df) != expected_rows:
        print(f"✗ Expected {expected_rows} data rows, got {len(df)}")
        return False
    print(f"✓ Correct number of data rows: {len(df)} (1+2+...+200)")
    
    # Check ID format
    bad_ids = []
    for idx, row in df.iterrows():
        tree_id = row['id']
        parts = tree_id.split('_')
        if len(parts) != 2:
            bad_ids.append(tree_id)
        else:
            try:
                n = int(parts[0])
                t = int(parts[1])
                if n < 1 or n > 200 or t < 0 or t >= n:
                    bad_ids.append(tree_id)
            except:
                bad_ids.append(tree_id)
    
    if bad_ids:
        print(f"✗ Invalid IDs: {bad_ids[:5]}")
        return False
    print("✓ All IDs valid (NNN_M format, 0 ≤ M < N)")
    
    return True


def validate_solution_feasibility(filepath: str) -> bool:
    """Validate solution feasibility."""
    print("\n[2] Solution Feasibility Validation")
    print("-" * 70)
    
    # Load solution
    df = pd.read_csv(filepath)
    solution = PackingSolution.from_dataframe(df)
    
    # Validate
    validator = SolutionValidator()
    is_valid, errors = validator.validate_solution(solution)
    
    if not is_valid:
        print(f"✗ Solution has {len(errors)} errors:")
        for err in errors[:5]:
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
        return False
    
    print("✓ No collisions detected")
    print("✓ All trees fit in squares")
    print("✓ All coordinates in bounds")
    
    return True


def validate_score(filepath: str) -> bool:
    """Validate and report score."""
    print("\n[3] Score Calculation")
    print("-" * 70)
    
    df = pd.read_csv(filepath)
    solution = PackingSolution.from_dataframe(df)
    
    score = solution.compute_score()
    print(f"✓ Total Score: {score:.10f}")
    
    # Per-configuration breakdown
    print("\n  Per-configuration contributions:")
    for n in [1, 10, 50, 100, 150, 200]:
        if n in solution.trees:
            from decimal import Decimal
            side = solution.get_square_side(n)
            contrib = (side ** 2) / Decimal(str(n))
            print(f"    n={n:3d}: side={float(side):8.6f}, contribution={float(contrib):8.6f}")
    
    return True


def main():
    """Main validation."""
    submission_file = Path('/home/engine/project/output/FINAL_SUBMISSION.csv')
    
    if not submission_file.exists():
        print(f"✗ Submission file not found: {submission_file}")
        return 1
    
    print("=" * 70)
    print("SANTA 2025 - SUBMISSION VALIDATOR")
    print("=" * 70)
    print(f"\nFile: {submission_file}")
    print(f"Size: {submission_file.stat().st_size / (1024*1024):.1f} MB")
    
    # Run validations
    checks = [
        ("CSV Format", validate_csv_format),
        ("Feasibility", validate_solution_feasibility),
        ("Score", validate_score),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func(str(submission_file))
            results.append((name, result))
        except Exception as e:
            print(f"✗ Exception in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:30s} {status}")
    
    all_pass = all(r for _, r in results)
    
    if all_pass:
        print("\n✓ SUBMISSION READY FOR COMPETITION!")
        return 0
    else:
        print("\n✗ SUBMISSION HAS ISSUES")
        return 1


if __name__ == '__main__':
    sys.exit(main())
