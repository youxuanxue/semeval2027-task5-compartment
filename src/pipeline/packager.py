"""
Submission Formatter and Packager for SemEval-2027 Task 5
Generates compliant predictions TSV and builds ZIP package for Codabench.
"""

import os
import zipfile
from typing import Dict, List, Tuple


def write_predictions(
    output_tsv: str,
    predictions: List[dict],
    task_type: str,
):
    """
    Format required:
    No header. Tab-separated values:
    - Noun compound: ContextID 	 ModPred 	 HeadPred
    - Particle verb: ContextID 	 Pred
    """
    with open(output_tsv, "w", encoding="utf-8") as f:
        for p in predictions:
            cid = p["ContextID"]
            if task_type == "nn":
                mod_p = f"{float(p['mod_pred']):.6f}"
                head_p = f"{float(p['head_pred']):.6f}"
                f.write(f"{cid}\t{mod_p}\t{head_p}\n")
            else:
                pred = f"{float(p['pred']):.6f}"
                f.write(f"{cid}\t{pred}\n")


def build_submission_zip(
    tsv_files: List[str],
    output_zip: str,
):
    """Packages the prediction TSVs into submission.zip"""
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in tsv_files:
            arcname = os.path.basename(path)
            zf.write(path, arcname=arcname)
    print(f"[+] Successfully built submission zip: {output_zip} ({os.path.getsize(output_zip)} bytes)")
