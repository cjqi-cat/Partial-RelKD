from pathlib import Path
import sys


def score(change_agent_root, references, hypotheses):
    root = str(Path(change_agent_root))
    if root not in sys.path: sys.path.insert(0, root)
    from eval_func.bleu.bleu import Bleu
    from eval_func.rouge.rouge import Rouge
    from eval_func.cider.cider import Cider
    gts = {i: [" ".join(map(str, x)) for x in group] for i, group in enumerate(references)}
    res = {i: [" ".join(map(str, row))] for i, row in enumerate(hypotheses)}
    bleu, _ = Bleu(4).compute_score(gts, res)
    rouge, _ = Rouge().compute_score(gts, res)
    cider, _ = Cider().compute_score(gts, res)
    return {"BLEU-1": float(bleu[0]), "BLEU-2": float(bleu[1]), "BLEU-3": float(bleu[2]),
            "BLEU-4": float(bleu[3]), "ROUGE-L": float(rouge), "CIDEr": float(cider)}
