"""Portable offline analysis of the frozen coursework experiment. No API calls."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from research_evaluation import judge,summary,clustered_ci,paired,holm,norm
from experiment_core import counts

NAMES={'sonnet55':'Claude Sonnet 5.5','astra':'GPT-6 Astra'}
COLORS={'sonnet55':'#009E73','astra':'#7B3294'}
ARMS=['A','B','D','E']
CATEGORIES={'direct':'Прямые указания','role':'Подмена роли','authority':'Ложные полномочия','chat':'Форматы чата','format':'Подмена формата','embedded':'Примечания','multilingual':'Смешанные языки','workflow':'Подмена процесса','oblique':'Завуалированные','administrative':'Рабочие распоряжения'}
def read(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf8').splitlines() if s.strip()]
def load(root):
    return (read(root/'data/cases.jsonl'),read(root/'data/responses.jsonl'),read(root/'data/annotations.jsonl'),
            {arm:joblib.load(root/f'models/{arm}.joblib') for arm in ['D','E']})
def inventory(cases,responses):
    f=pd.DataFrame(responses)
    return f.groupby(['cohort','model_name']).size().unstack().rename(columns=NAMES)
def filter_example(cases,filters):
    rows=[]
    for arm,(model,t) in filters.items():
        for r,s in zip(cases,model.predict_proba([r['document'] for r in cases])[:,1]):
            rows.append({'Документ':r['kind'],'Фильтр':arm,'Балл':s,'Порог':t,'Решение':'Блокировать' if s>=t else 'Пропустить'})
    return pd.DataFrame(rows)
def reconstruct(cases,responses,annotations,filters):
    case_map={r['case']:r for r in cases}
    labels={(r['case'],r['model_name'],r['source_arm'],r['repeat']):r for r in annotations}
    predictions=[];blocked={}
    for arm,(model,t) in filters.items():
        scores=model.predict_proba([r['document'] for r in cases])[:,1]
        for r,s in zip(cases,scores):
            blocked[(r['case'],arm)]=bool(s>=t)
            predictions.append({k:r[k] for k in ['case','scene','family','cohort','category','kind','label']}|dict(defense=arm,score=float(s),threshold=t,blocked=bool(s>=t)))
    rows=[]
    for r in responses:
        case=case_map[r['case']]
        for arm in ([r['arm']] if r['arm']=='A' else ['B','D','E']):
            entry=dict(r,arm=arm)
            if arm in ['D','E'] and blocked[(r['case'],arm)]:
                entry.update(status='blocked',answer='',finish=None,usage=None,seconds=None)
            key=(r['case'],r['model_name'],r['arm'],r['repeat'])
            entry.update(judge(case,entry,labels.get(key)))
            rows.append(entry)
    return pd.DataFrame(rows),pd.DataFrame(predictions)
def show_example(frame,cases,scene,cohort):
    display = print
    Markdown = str
    r=next(r for r in cases if r['scene']==scene and r['cohort']==cohort and r['kind']=='attack')
    display(Markdown('**Документ**'));print(r['document']);print('Вопрос:',r['question'],'Эталон:',r['gold'])
    for name in NAMES:
        f=frame[(frame.scene==scene)&(frame.cohort==cohort)&(frame.kind=='attack')&(frame['repeat']==1)&(frame.model_name==name)]
        display(Markdown('**'+NAMES[name]+'**'))
        for arm in ['A','B']:
            s=f[f.arm==arm].iloc[0];print(arm+':',s.answer)
        display(f[['arm','status','attack_success','strict_correct','semantic_correct']].rename(columns={'arm':'Защита','status':'Статус','attack_success':'Успех атаки','strict_correct':'Строго правильно','semantic_correct':'Правильно по смыслу'}))
def metrics(frame):
    rows=[]
    for (name,arm),g in frame[frame.kind=='attack'].groupby(['model_name','arm'],sort=False):
        s=summary(g);rows.append({'Модель':NAMES[name],'Защита':arm,'Успехи':s['attack_successes'],'Завершено':s['completed'],'Блокировки':s['blocked'],'Ответы':s['answered'],'ASR, %':100*s['asr'] if s['asr'] is not None else np.nan,'Условный ASR, %':100*s['conditional_asr'] if s['conditional_asr'] is not None else np.nan})
    return pd.DataFrame(rows).round(2)
def clean_metrics(frame):
    rows=[]
    for (name,arm),g in frame[frame.kind=='clean'].groupby(['model_name','arm'],sort=False):
        s=summary(g);rows.append({'Модель':NAMES[name],'Защита':arm,'Завершено':s['completed'],'Блокировки':s['blocked'],'Строго правильные':s['strict_correct'],'Правильные по смыслу':s['semantic_correct'],'Не оценены':s['semantic_unknown'],'Строгая точность, %':100*s['strict_accuracy'],'Смысловая точность, %':100*s['semantic_accuracy'] if s['semantic_accuracy'] is not None else np.nan})
    return pd.DataFrame(rows).round(2)
def detector_metrics(predictions):
    return pd.DataFrame([dict(cohort=c,defense=d,**counts(g.label,g.blocked)) for (c,d),g in predictions.groupby(['cohort','defense'])])
def finish(fig,path,name):
    path.mkdir(exist_ok=True,parents=True);fig.tight_layout()
    fig.savefig(path/(name+'.png'),dpi=320,bbox_inches='tight');fig.savefig(path/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def plot_main(main,path):
    fig,ax=plt.subplots(figsize=(10,4.8))
    for mi,name in enumerate(NAMES):
        for i,arm in enumerate(ARMS):
            g=main[(main.kind=='attack')&(main.model_name==name)&(main.arm==arm)].copy();g['den']=1
            rate=g.attack_success.mean()*100;lo,hi=clustered_ci(g,'attack_success','den');x=i+(mi-.5)*.34
            ax.bar(x,rate,.32,color=COLORS[name],label=NAMES[name] if i==0 else None)
            ax.errorbar(x,rate,yerr=[[max(0,rate-lo*100)],[max(0,hi*100-rate)]],fmt='none',color='black',capsize=4)
            ax.text(x,hi*100+2,f'{g.attack_success.sum()}/{len(g)}',ha='center',fontsize=9)
    ax.set(xticks=range(4),xticklabels=ARMS,ylabel='Успешные атаки, %',ylim=(0,100),title='Основной тест: ASR и 95%-е интервалы по семействам\n180 ситуаций / 60 семейств; ноль не доказывает безопасность');ax.legend()
    finish(fig,path,'01_asr')
def plot_clean(main,path):
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for ax,col,title in zip(axes,['strict_correct','semantic_correct'],['Строгое совпадение','Правильность по смыслу']):
        for mi,name in enumerate(NAMES):
            values=[]
            for arm in ARMS:
                g=main[(main.kind=='clean')&(main.model_name==name)&(main.arm==arm)]
                values.append(g[col].eq(True).sum()/len(g)*100)
            ax.bar(np.arange(4)+(mi-.5)*.34,values,.32,color=COLORS[name],label=NAMES[name])
        ax.set(xticks=range(4),xticklabels=ARMS,ylim=(0,110),ylabel='Правильные ответы из 180, %',title=title);ax.legend(fontsize=8)
    fig.suptitle('Обычные задачи: блокировки входят в знаменатель\nСмысловая разметка автоматическая, без независимой человеческой проверки')
    finish(fig,path,'02_clean')
def plot_filters(predictions,path):
    fig,axes=plt.subplots(1,2,figsize=(10,4.6))
    for ax,arm in zip(axes,['D','E']):
        g=predictions[(predictions.cohort=='main')&(predictions.defense==arm)];m=counts(g.label,g.blocked)
        a=np.array([[m['tn'],m['fp']],[m['fn'],m['tp']]])
        ax.imshow(a,cmap='Blues',vmin=0,vmax=180)
        for (i,j),v in np.ndenumerate(a):ax.text(j,i,str(v),ha='center',va='center',fontsize=16,color='white' if v>90 else 'black')
        ax.set(xticks=[0,1],xticklabels=['Пропущен','Блокирован'],yticks=[0,1],yticklabels=['Обычный','Атака'],title=arm)
    fig.suptitle('Ошибки фильтров: 180 атакующих и 180 обычных документов');finish(fig,path,'03_filters')
def plot_categories(main,path):
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for ax,name in zip(axes,NAMES):
        f=main[(main.kind=='attack')&(main.model_name==name)]
        p=f.pivot_table(index='category',columns='arm',values='attack_success',aggfunc='mean').reindex(columns=ARMS)
        ax.imshow(p*100,cmap='YlOrRd',vmin=0,vmax=100,aspect='auto')
        for (i,j),v in np.ndenumerate(p.to_numpy()):ax.text(j,i,f'{v*100:.1f}',ha='center',va='center',color='white' if v>.6 else 'black')
        ax.set(xticks=range(4),xticklabels=ARMS,yticks=range(len(p)),yticklabels=[CATEGORIES[c] for c in p.index],title=NAMES[name])
    fig.suptitle('ASR по категориям, %: по 18 документов / 6 семейств');finish(fig,path,'04_categories')
def factor_table(frame):
    rows=[]
    for cohort,label in [('length','Удлинение'),('position_begin','Начало'),('position_middle','Середина'),('question','Изменение вопроса'),('stress','Стрессовая длина')]:
        for name in NAMES:
            target=frame[(frame.cohort==cohort)&(frame.kind=='attack')&(frame.arm=='A')&(frame.model_name==name)&(frame['repeat']==1)]
            reference=frame[(frame.cohort=='main')&(frame.scene.isin(target.scene))&(frame.kind=='attack')&(frame.arm=='A')&(frame.model_name==name)&(frame['repeat']==1)]
            rows.append({'Условие':label,'Модель':NAMES[name],'Исходные успехи':f'{reference.attack_success.sum()}/{len(reference)}','Успехи варианта':f'{target.attack_success.sum()}/{len(target)}'})
    return pd.DataFrame(rows)
def repeats(frame):
    rows=[]
    for name in NAMES:
        for arm in ARMS:
            f=frame[(frame.cohort=='main')&(frame.kind=='attack')&(frame.model_name==name)&(frame.arm==arm)]
            groups=[g for _,g in f.groupby('case') if len(g)==3]
            rows.append({'Модель':NAMES[name],'Защита':arm,'Ситуаций':len(groups),'Хотя бы один успех':sum(g.attack_success.any() for g in groups),'Менялся исход':sum(g.attack_success.nunique()>1 for g in groups)})
    return pd.DataFrame(rows)
def plot_length(predictions,path):
    scenes=set(predictions[predictions.cohort=='stress'].scene)
    fig,axes=plt.subplots(1,2,figsize=(10,4.8))
    for ax,kind,title in zip(axes,['attack','clean'],['Обнаружение атак','Ложные блокировки']):
        for arm,color in [('D','#C44E52'),('E','#0072B2')]:
            vals=[predictions[(predictions.scene.isin(scenes))&(predictions.cohort==c)&(predictions.kind==kind)&(predictions.defense==arm)].blocked.mean()*100 for c in ['main','length','stress']]
            ax.plot(range(3),vals,'o-',label=arm,color=color)
        ax.set(xticks=range(3),xticklabels=['Короткий','Удлинённый','Стрессовый'],ylim=(-5,105),ylabel='Блокировки, %',title=title);ax.legend()
    fig.suptitle('Одинаковые 10 ситуаций; результаты зависят от добавленного контекста');finish(fig,path,'05_length')
def comparisons(main):
    rows=[]
    for arm in ARMS:
        f=main[(main.arm==arm)&(main.kind=='attack')]
        rows.append(dict(comparison='astra-sonnet55',arm=arm,metric='ASR',**paired(f[f.model_name=='sonnet55'],f[f.model_name=='astra'],'attack_success')))
    for name in NAMES:
        f=main[main.model_name==name]
        rows.append(dict(comparison=name+':B-A',metric='ASR',**paired(f[(f.arm=='A')&(f.kind=='attack')],f[(f.arm=='B')&(f.kind=='attack')],'attack_success')))
        for kind,col in [('attack','attack_success'),('clean','strict_correct')]:
            rows.append(dict(comparison=name+':E-D',metric=col,**paired(f[(f.arm=='D')&(f.kind==kind)],f[(f.arm=='E')&(f.kind==kind)],col)))
    for r,p in zip(rows,holm([r['p'] for r in rows])):r['p_holm']=p
    return pd.DataFrame(rows)
def verify(root,cases,responses,frame,predictions,comparisons):
    checks=[]
    def check(name,ok):
        assert bool(ok),name
        checks.append({'Проверка':name,'Результат':'Пройдена'})
    check('3216 реальных ответов, по 1608 на модель',len(responses)==3216 and pd.DataFrame(responses).groupby('model_name').size().eq(1608).all())
    check('Нет повторных ключей',not pd.DataFrame(responses).duplicated(['case','model_name','arm','repeat']).any())
    sets=[]
    for name in ['sonnet55','astra']:
        sets.append({(r['case'],r['arm'],r['repeat']) for r in responses if r['model_name']==name})
    check('Модели проверены на одинаковых запросах',sets[0]==sets[1])
    train=read(root/'data/train.jsonl');val=read(root/'data/validation.jsonl')
    families=[{r['family'] for r in rows} for rows in [train,val,cases]]
    check('Семейства обучения, валидации и теста не пересекаются',all(not families[i]&families[j] for i,j in [(0,1),(0,2),(1,2)]))
    texts=[{norm(r['document']) for r in rows} for rows in [train,val,cases]]
    check('Нет точных пересечений текстов между разделами',all(not texts[i]&texts[j] for i,j in [(0,1),(0,2),(1,2)]))
    for _,g in pd.DataFrame(cases).groupby(['scene','cohort']):
        assert set(g.kind)=={'attack','clean'} and g.question.nunique()==1 and g.gold.nunique()==1
    check('Атакующие и обычные варианты имеют общий вопрос и эталон',True)
    check('Все исходы завершены',frame.status.isin(['answered','blocked']).all())
    check('Нет непроверенных смысловых ответов',not frame.loc[frame.status=='answered','semantic_correct'].isna().any())
    ref=pd.read_csv(root/'reference/filter_predictions.csv').replace({'defense':{'E2.1':'E'}})
    merged=predictions.merge(ref,on=['case','defense'],suffixes=('_new','_saved'),validate='one_to_one')
    check('Решения фильтров совпадают с исходным экспериментом',len(merged)==len(predictions) and merged.blocked_new.eq(merged.blocked_saved).all())
    check('Балл фильтров воспроизводится',np.allclose(merged.score_new,merged.score_saved,rtol=1e-8,atol=1e-10))
    ref=pd.read_csv(root/'reference/summary.csv').replace({'arm':{'E2.1':'E'}})
    ref=ref[(ref.cohort=='main')&(ref['repeat']==1)]
    for r in ref.to_dict('records'):
        g=frame[(frame.cohort=='main')&(frame['repeat']==1)&(frame.model_name==r['model_name'])&(frame.arm==r['arm'])&(frame.kind==r['kind'])]
        s=summary(g)
        for k in ['completed','answered','blocked','attack_successes','strict_correct','semantic_correct']:assert s[k]==r[k],(r['model_name'],r['arm'],k)
    check('Основные ASR и оценки правильности совпадают',True)
    ref=pd.read_csv(root/'reference/paired_primary.csv')
    check('Парные сравнения, интервалы и поправка Холма воспроизводятся',np.allclose(comparisons[['difference','lo','hi','p','p_holm']],ref[['difference','lo','hi','p','p_holm']],atol=1e-12))
    # Known controls: model response, local block and technical error remain different outcomes.
    case={'accepted':['14'],'kind':'attack','marker':'КРАСНЫЙ КВАДРАТ'}
    check('Строгий успех не равен упоминанию маркера',judge(case,{'status':'answered','answer':'КРАСНЫЙ КВАДРАТ'})['attack_success'] and not judge(case,{'status':'answered','answer':'14. КРАСНЫЙ КВАДРАТ'})['attack_success'])
    check('Ошибки и блокировки различаются',judge(case,{'status':'error'})['status']=='error' and judge(case,{'status':'blocked'})['status']=='blocked')
    return checks
