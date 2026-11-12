import argparse,torch
from common import load_json
from data import CaptionPairs,CaptionTokens
from model import build_model
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/LEVIR_MCI/rmnet.json'); a=p.parse_args(); c=load_json(a.config); d=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); t=CaptionTokens(c['metadata_root'],c['max_length']); ds=CaptionPairs(c,'train',t,True); b=ds[0]; m=build_model(c,t.vocab).to(d); out=m(b['a'][None].to(d),b['b'][None].to(d),b['tokens'][None].to(d),b['length'][None].to(d)); out[0].mean().backward(); m.eval(); seq=m.greedy(b['a'][None].to(d),b['b'][None].to(d)); print({'model':'RMNet','scores':tuple(out[0].shape),'sequence':tuple(seq.shape),'trainable':sum(x.numel() for x in m.parameters() if x.requires_grad)}); print('SMOKE_OK')
