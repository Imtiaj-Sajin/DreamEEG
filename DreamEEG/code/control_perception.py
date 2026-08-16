"""
Control experiment on sub-09 (animals):
Decode the PERCEPTION window (subject actually viewing the image) vs the
IMAGERY window (subject imagining it), using the SAME pipeline.

Trial: fixation 3s -> image 4s -> mask 2s -> imagery 4s -> rest.
The single trigger marks imagery onset (verified against events.tsv).
So relative to the trigger:
   imagery window  = 0 .. 4 s
   perception (image-viewing) window = -6 .. -2 s
If perception decodes well but imagery does not, the pipeline is sound and
this subject's imagery signal is simply weak.
"""
import glob, warnings, numpy as np, mne, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")
SEED=42; FS=250; np.random.seed(SEED); torch.manual_seed(SEED)
EVENT_ID={"dog":1,"bird":2,"fish":3}

def events_from_status(raw):
    st=raw.get_data(picks=["Status"])[0]; v,c=np.unique(st,return_counts=True); b=v[np.argmax(c)]
    idx=np.where(np.isin(st,[1,2,3]))[0]; ev=[]
    for i in idx:
        p=st[i-1] if i>0 else b; n=st[i+1] if i<len(st)-1 else b
        if p==b and n==b: ev.append([i,0,int(st[i])])
    return np.array(ev,dtype=int)

def load_raw():
    bdf=glob.glob("../data/sub-09/sub-09/ses-01/eeg/*task-AVI*eeg.bdf")[0]
    raw=mne.io.read_raw_bdf(bdf,preload=True)
    ev=events_from_status(raw); raw._data*=1e-6
    raw,ev=raw.resample(FS,events=ev,verbose=False)
    raw.set_channel_types({"Status":"stim"})
    raw.filter(1.,40.,method="fir",phase="zero-double",pad="edge",verbose=False)
    raw.set_eeg_reference("average",verbose=False); raw.pick("eeg")
    return raw,ev

def epoch(raw,ev,tmin,tmax):
    ep=mne.Epochs(raw,ev,event_id=EVENT_ID,tmin=tmin,tmax=tmax,baseline=None,
                  preload=True,on_missing="warn",verbose=False)
    X=ep.get_data()[:,:32,:1000].astype(np.float32); y=ep.events[:,2]-1
    return X,y

class EEGNet(nn.Module):
    def __init__(s,ch=32,cl=3,tp=1000,tk=25,f1=8,f2=16,d=2,pk1=16,pk2=8,dr=0.5):
        super().__init__(); lin=(tp//(pk1*pk2))*f2
        s.b1=nn.Sequential(nn.Conv2d(1,f1,(1,tk),padding="same",bias=False),nn.BatchNorm2d(f1))
        s.b2=nn.Sequential(nn.Conv2d(f1,d*f1,(ch,1),groups=f1,bias=False),nn.BatchNorm2d(d*f1),nn.ELU(),nn.AvgPool2d((1,pk1)),nn.Dropout(dr))
        s.b3=nn.Sequential(nn.Conv2d(d*f1,f2,(1,16),groups=f2,bias=False,padding="same"),nn.Conv2d(f2,f2,1,bias=False),nn.BatchNorm2d(f2),nn.ELU(),nn.AvgPool2d((1,pk2)),nn.Dropout(dr))
        s.h=nn.Sequential(nn.Flatten(),nn.Linear(lin,cl))
    def forward(s,x): return s.h(s.b3(s.b2(s.b1(x))))

def cv(X,y,epochs=300,tag=""):
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf=StratifiedKFold(5,shuffle=True,random_state=SEED); accs=[]
    for tr,va in skf.split(X,y):
        Xtr,Xva=X[tr],X[va]; m=Xtr.mean((0,2),keepdims=True); sd=Xtr.std((0,2),keepdims=True)+1e-8
        Xtr,Xva=(Xtr-m)/sd,(Xva-m)/sd
        tl=DataLoader(TensorDataset(torch.tensor(Xtr[:,None]),torch.tensor(y[tr])),64,shuffle=True)
        vl=DataLoader(TensorDataset(torch.tensor(Xva[:,None]),torch.tensor(y[va])),128)
        net=EEGNet().to(dev); opt=optim.Adam(net.parameters(),lr=1e-3,weight_decay=0.09); crit=nn.CrossEntropyLoss()
        best=0.
        for _ in range(epochs):
            net.train()
            for xb,yb in tl:
                xb,yb=xb.to(dev),yb.to(dev); opt.zero_grad(); crit(net(xb),yb).backward(); opt.step()
            net.eval(); P,Y=[],[]
            with torch.no_grad():
                for xb,yb in vl: P+=net(xb.to(dev)).argmax(1).cpu().tolist(); Y+=yb.tolist()
            best=max(best,float(np.mean(np.array(P)==np.array(Y))))
        accs.append(best)
    print(f"  {tag}: {np.mean(accs)*100:.1f}% +/- {np.std(accs)*100:.1f}%  (chance 33.3%)")
    return np.mean(accs)

if __name__=="__main__":
    raw,ev=load_raw()
    print("[perception] image-viewing window (trigger -6..-2 s):")
    Xp,yp=epoch(raw,ev,-6.0,-2.0); cv(Xp,yp,tag="PERCEPTION")
    print("[imagery] imagery window (trigger 0..4 s):")
    Xi,yi=epoch(raw,ev,0.0,4.0);   cv(Xi,yi,tag="IMAGERY   ")
