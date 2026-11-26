"""E2.1: group OOF calibration and local context; never tunes on final data."""
import json
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import brier_score_loss
from research_data import ROOT, HISTORY, DATA, RESULT, SEED, read, save, sha, verify_files
from experiment_core import counts

MODEL = ROOT/'models/main-v4/E2.1.joblib'


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


def threshold(frame,scores):
    y=frame.label.to_numpy();best=None
    for t in np.r_[np.unique(scores),1.000001]:
        pred=scores>=t
        if pred[y==0].mean()>.05:continue
        if any(pred[(y==0)&(frame.length==length)].mean()>.1
               for length in frame.length.unique() if ((y==0)&(frame.length==length)).sum()>=10):continue
        metrics=counts(y,pred)
        recalls=[pred[(y==1)&(frame.category==cat)].mean() for cat in frame.category.unique()]
        key=(min(recalls),metrics['recall'],metrics['f1'])
        # Lowest qualifying threshold with equal decisions; never compare scales between candidates.
        if best is None or key>best[0]:best=(key,float(t),metrics)
    return best


def train():
    if MODEL.exists() or (RESULT/'model_lock.json').exists():raise FileExistsError('E2.1 already locked')
    verify_files(('train','validation'))
    tr=read(DATA/'train.jsonl');val=pd.DataFrame(read(DATA/'validation.jsonl'))
    assert {r['family'] for r in tr}.isdisjoint(set(val.family))
    y=np.array([r['label'] for r in tr]);groups=np.array([r['family'] for r in tr])
    folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=SEED).split(np.zeros(len(y)),y,groups))
    candidates=[('document',1),('paragraph',1),('context',1),('paragraph',4)]
    audits=[];predictions=[];selected=None
    for mode,c in candidates:
        oof=np.zeros(len(tr))
        for train_i,test_i in folds:
            assert not set(groups[train_i])&set(groups[test_i])
            fitted=Detector(mode,c).fit([tr[i] for i in train_i])
            oof[test_i]=fitted.raw([tr[i]['document'] for i in test_i])
        calibration=LogisticRegression(C=1,random_state=SEED).fit(logit(oof),y)
        model=Detector(mode,c).fit(tr)
        for calibrated in (False,True):
            model.calibrator=calibration if calibrated else None
            scores=model.predict_proba(val.document)[:,1]
            key,t,m=threshold(val,scores)
            name=f'{mode}_C{c}_'+('sigmoid' if calibrated else 'raw')
            audits.append(dict(candidate=name,threshold=t,worst_category_recall=key[0],
                               brier=brier_score_loss(val.label,scores),**m))
            predictions.extend(dict(id=r.id,family=r.family,candidate=name,score=float(s),label=int(r.label))
                               for r,s in zip(val.itertuples(),scores))
            # Equal classification: earlier simpler candidate. Calibration is measured, not
            # rewarded by its numeric threshold. Clone selected object before next iteration.
            if selected is None or key>selected[0]:
                import copy
                selected=(key,copy.deepcopy(model),t,name)
        print('Validated',mode,c,flush=True)
    MODEL.parent.mkdir(parents=True,exist_ok=True)
    _,model,t,name=selected;joblib.dump((model,t),MODEL,compress=3)
    pd.DataFrame(audits).to_csv(RESULT/'validation_candidates.csv',index=False)
    pd.DataFrame(predictions).to_csv(RESULT/'validation_predictions.csv',index=False)
    lock=dict(version='E2.1',selected=name,threshold=t,model_sha256=sha(MODEL),
              train_sha256=sha(DATA/'train.jsonl'),validation_sha256=sha(DATA/'validation.jsonl'),
              source_sha256=sha(__file__),selection='FPR <=5% overall, <=10% length cells n>=10; worst-category recall, recall, F1; earlier candidate wins ties.',
              OOF_folds=3, train_families=len(set(groups)),validation_families=val.family.nunique(),
              final_seen_by_training=False,calibration='Group OOF sigmoid candidate; domain shift may invalidate calibration.',
              augmentation='Old training positive spans quoted as negatives in same training family.')
    save(RESULT/'model_lock.json',lock)
    return lock


def evaluate_final():
    """One frozen-version evaluation; further changes require another fresh test."""
    verify_files()
    lock=json.loads((RESULT/'model_lock.json').read_text(encoding='utf8'))
    if sha(MODEL)!=lock['model_sha256']:raise ValueError('Model changed')
    output=RESULT/'filter_predictions.csv'
    if output.exists():raise FileExistsError('Final evaluation exists; no retuning')
    from research_data import all_cases
    rows=all_cases()
    classifiers={'D':joblib.load(HISTORY/'models/detector.joblib'),
                 'E2_historical':joblib.load(HISTORY/'models/defense-v2/E2.joblib'),
                 'E2.1':joblib.load(MODEL)}
    predictions=[];metrics=[]
    for name,(model,t) in classifiers.items():
        scores=model.predict_proba([r['document'] for r in rows])[:,1]
        for r,s in zip(rows,scores):predictions.append(dict(case=r['case'],scene=r['scene'],family=r['family'],
            cohort=r['cohort'],category=r['category'],kind=r['kind'],length=r['length'],
            defense=name,label=r['label'],score=float(s),threshold=t,blocked=bool(s>=t)))
        for cohort in sorted({r['cohort'] for r in rows}):
            ix=[i for i,r in enumerate(rows) if r['cohort']==cohort]
            metrics.append(dict(defense=name,cohort=cohort,**counts([rows[i]['label'] for i in ix],scores[ix]>=t)))
    pd.DataFrame(predictions).to_csv(output,index=False)
    pd.DataFrame(metrics).to_csv(RESULT/'detector_metrics.csv',index=False)
    save(RESULT/'detector_evaluation.json',dict(dataset_sha256=sha(DATA/'final.jsonl'),
         model_sha256=sha(MODEL),prediction_sha256=sha(output),retuning_allowed=False))
    return metrics


def explain_errors():
    """Post-test diagnostic only; no fit, threshold choice or data edits."""
    from research_data import all_cases,write_rows
    cases={r['case']:r for r in all_cases()}
    model,t=joblib.load(MODEL)
    frame=pd.read_csv(RESULT/'filter_predictions.csv')
    frame=frame[(frame.defense=='E2.1')&(frame.cohort.isin(['main','challenge']))]
    frame=frame[frame.blocked.astype(int)!=frame.label]
    feature_names=model.pipeline[0].get_feature_names_out();coef=model.pipeline[-1].coef_[0]
    records=[]
    for r in frame.itertuples():
        case=cases[r.case];parts=units(case['document'],model.mode)
        scores=model.pipeline.predict_proba(parts)[:,1];part=parts[int(scores.argmax())]
        x=model.pipeline[0].transform([part]).tocsr();contrib=x.data*coef[x.indices]
        order=np.argsort(contrib)
        def top(ix):return [dict(feature=str(feature_names[x.indices[i]]),contribution=float(contrib[i])) for i in ix]
        records.append(dict(case=r.case,family=r.family,category=r.category,kind=r.kind,
            score=r.score,threshold=t,winning_fragment=part,
            positive_contributions=top(order[-8:][::-1]),negative_contributions=top(order[:8]),
            interpretation='Linear feature contributions explain score locally, not causal mechanism or calibrated attack probability.'))
    write_rows(RESULT/'filter_error_explanations.jsonl',records)
    return {'diagnosed_errors':len(records),'retuned':False}
