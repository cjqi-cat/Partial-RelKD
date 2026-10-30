#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

P=Path('__PROJECT_ROOT__');SRC=P/'stage10_7_severity_curves/results';OUT=P/'stage10_8_main_completion/results/qualitative';OUT.mkdir(parents=True,exist_ok=True)
vocab=json.loads((P/'stage3_caption_retrain/vendor/Change-Agent/Multi_change/data/LEVIR_MCI/vocab.json').read_text());inv={int(v):k for k,v in vocab.items()}
def text(ids):return ' '.join(inv.get(int(x),'<UNK>') for x in ids)
def sim(pred,refs):
 a=set(pred);return max((2*len(a&set(r))/max(1,len(a)+len(set(r))) for r in refs),default=0)
def main():
 cases=[]
 for cond in ('blur__both','low_resolution__both','misregistration__both','mixed__both'):
  rows={m:{x['name']:x for x in json.loads((SRC/m/'severity_5'/f'{cond}.predictions.json').read_text())} for m in ('change_agent','p3','p3_nomask')};names=sorted(rows['p3_nomask']);rank=[]
  for n in names:
   rr=rows['p3_nomask'][n]['reference_ids'];scores={m:sim(rows[m][n]['prediction_ids'],rr) for m in rows};rank.append((scores['p3_nomask']-max(scores['p3'],scores['change_agent']),n,scores,rr))
  picks=sorted(rank,reverse=True)[:2]+sorted(rank)[:1]
  for delta,n,scores,rr in picks:cases.append({'condition':cond,'name':n,'delta':delta,'reference':text(rr[0]),**{m:text(rows[m][n]['prediction_ids']) for m in rows}})
 (OUT/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8');lines=['# Severity-5定性案例（按token-F1自动选择）','', '> 每个条件选P3-NoMask优势最大的2例和劣势最大的1例；用于发现模式，不用于统计结论。','']
 for i,c in enumerate(cases,1):lines += [f"## {i}. {c['condition']} / {c['name']}",'',f"- Reference: {c['reference']}",f"- Change-Agent: {c['change_agent']}",f"- P3: {c['p3']}",f"- P3-NoMask: {c['p3_nomask']}",f"- NoMask token-F1 advantage: {c['delta']:+.4f}",'']
 (OUT/'qualitative_cases.md').write_text('\n'.join(lines),encoding='utf-8');print('QUALITATIVE_OK')
if __name__=='__main__':main()
