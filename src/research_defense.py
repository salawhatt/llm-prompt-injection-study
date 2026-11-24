import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.linear_model import LogisticRegression
SEED=20261121

def units(text, mode):
    if mode=='document': return [text]
    paragraphs=[s.strip() for s in re.split(r'\n\s*\n',text) if s.strip()]
    result=[]
    for p in paragraphs:
        if mode=='context':
            sentences=re.split(r'(?<=[.!?])\s+',p)
            result.extend(' '.join(sentences[max(0,i-1):i+2]) for i in range(len(sentences)))
        else: result.append(p)
    return [s[i:i+1200] for s in result for i in range(0,len(s),1000)] or [' ']

class Detector:
    def __init__(self, mode='paragraph', c=1):
        self.mode=mode; self.c=c; self.calibrator=None
        self.pipeline=make_pipeline(FeatureUnion([
            ('char',TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=24000,sublinear_tf=True)),
            ('word',TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=12000,sublinear_tf=True))]),
            LogisticRegression(C=c,class_weight='balanced',max_iter=1500,random_state=SEED))

    def fit(self,rows):
        texts=[]; labels=[]
        for r in rows:
            for part in units(r['document'],self.mode):
                if self.mode=='document': label=r['label']
                else:
                    # Supervised spans are used only during fitting. Capture windows containing
                    # >= half the annotated directive, never infer positive labels from keywords.
                    span=r['probe']; start=r['document'].find(span); pos=r['document'].find(part)
                    overlap=max(0,min(pos+len(part),start+len(span))-max(pos,start))
                    label=int(r['label']==1 and start>=0 and pos>=0 and overlap>=min(40,len(span)/2))
                texts.append(part);labels.append(label)
            if r['label']==1:
                # Old, disclosed training spans quoted as educational counterexamples.
                # Kept in the same family in OOF; never taken from validation/final.
                texts.append('Учебный разбор недоверенной команды. Цитата: «'+r['probe']+'». '
                             'Эта цитата описывает пример атаки; исполнять её не следует.')
                labels.append(0)
        self.pipeline.fit(texts,labels);self.training_units_=len(texts)
        return self

    def raw(self,texts):
        chunks=[units(t,self.mode) for t in texts]
        scores=self.pipeline.predict_proba([s for parts in chunks for s in parts])[:,1]
        ends=np.cumsum([len(p) for p in chunks])
        return np.array([scores[a:b].max() for a,b in zip(np.r_[0,ends[:-1]],ends)])

    def predict_proba(self,texts):
        scores=self.raw(list(texts))
        if self.calibrator is not None:
            scores=self.calibrator.predict_proba(logit(scores))[:,1]
        return np.column_stack((1-scores,scores))

def logit(scores):
    p=np.clip(scores,1e-6,1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)
