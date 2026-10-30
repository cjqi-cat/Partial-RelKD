#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

SPECIAL = ["<NULL>", "<UNK>", "<START>", "<END>"]
SPLITS = ("train", "val", "test")

def image_path(root, layout, split, side, name):
    return root / split / "rgb" / side / name if layout == "secondcc" else root / "images" / split / side / name

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--name",required=True)
    ap.add_argument("--dataset-root",type=Path,required=True); ap.add_argument("--annotation",type=Path,required=True)
    ap.add_argument("--layout",choices=("secondcc","standard"),required=True); ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--expected",default=""); ap.add_argument("--max-length",type=int,default=41); ap.add_argument("--require-label",action="store_true")
    a=ap.parse_args(); data=json.loads(a.annotation.read_text(encoding="utf-8")); images=data.get("images")
    if not isinstance(images,list): raise RuntimeError("annotation must contain images[]")
    grouped={s:[] for s in SPLITS}; vocab=Counter(); seen=set(); lengths=[]
    for item in images:
        split=str(item.get("split","")).lower()
        if split not in grouped: continue
        name=str(item["filename"])
        if name in seen: raise RuntimeError(f"duplicate filename: {name}")
        seen.add(name); captions=[]
        for sentence in item.get("sentences",[]):
            words=[str(w).lower() for w in sentence.get("tokens",[])]
            if words:
                captions.append(words); lengths.append(len(words))
                if split=="train": vocab.update(words)
        if not captions: raise RuntimeError(f"no captions: {name}")
        if max(map(len,captions))+2>a.max_length: raise RuntimeError(f"caption too long: {name}")
        for side in ("A","B"):
            path=image_path(a.dataset_root,a.layout,split,side,name)
            if not path.is_file(): raise FileNotFoundError(path)
        if a.require_label and not (a.dataset_root/"images"/split/"label"/name).is_file():
            raise FileNotFoundError(a.dataset_root/"images"/split/"label"/name)
        grouped[split].append((name,captions))
    counts=tuple(len(grouped[s]) for s in SPLITS)
    if a.expected and counts!=tuple(map(int,a.expected.split(","))): raise RuntimeError(f"split mismatch: {counts}")
    a.output.mkdir(parents=True,exist_ok=True); token_root=a.output/"tokens"; token_root.mkdir(exist_ok=True)
    for split in SPLITS:
        (a.output/f"{split}.txt").write_text("\n".join(n for n,_ in grouped[split])+"\n",encoding="utf-8")
        for name,captions in grouped[split]:
            encoded=[["<START>",*words,"<END>"] for words in captions]
            (token_root/f"{Path(name).stem}.txt").write_text(json.dumps(encoded,ensure_ascii=False),encoding="utf-8")
    mapping={t:i for i,t in enumerate(SPECIAL)}
    for word in sorted(vocab):
        if word not in mapping: mapping[word]=len(mapping)
    (a.output/"vocab.json").write_text(json.dumps(mapping,indent=2,ensure_ascii=False),encoding="utf-8")
    stats={"dataset":a.name,"split_counts":dict(zip(SPLITS,counts)),"vocabulary":len(mapping),"max_caption_length":max(lengths)}
    (a.output/"stats.json").write_text(json.dumps(stats,indent=2),encoding="utf-8"); print("METADATA_PREPARED",a.name)

if __name__=="__main__": main()
