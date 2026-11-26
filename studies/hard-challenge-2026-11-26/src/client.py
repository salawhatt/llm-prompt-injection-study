import hashlib
import json
import os
import sqlite3
import time
import httpx

URL = "https://foundation-models.api.cloud.ru/v1/chat/completions"

def usage_cost(cfg,usage):
    base=(usage['prompt_tokens']*cfg['input_rub_per_million']+usage['completion_tokens']*cfg['output_rub_per_million'])/1e6
    details=usage.get('prompt_tokens_details') or {}
    written=max(details.get('cache_write_tokens',0),details.get('cache_creation_tokens',0),details.get('cached_creation_tokens',0))
    if type(written) is not int or not 0<=written<=usage['prompt_tokens']:
        raise ValueError('Invalid cache usage')
    # Cloud.ru reports cache creation within prompt_tokens. Duration may be
    # missing, so use the published 1h upper tariff for Claude. No read discounts.
    write_rate=cfg.get('cache_write_rub_per_million',{'anthropic/claude-sonnet-5.5':711.97,'openai/gpt-6.1-sol':444.98,'openai/gpt-6-astra':2224.9,'anthropic/claude-opus-5.5':1423.94}.get(cfg['model'],cfg['input_rub_per_million']))
    return base+written*max(0,write_rate-cfg['input_rub_per_million'])/1e6

def identity(cfg, tag, messages):
    payload=dict(model=cfg["model"],messages=messages,max_tokens=cfg["max_output"],temperature=0)
    if 'temperature' in cfg:
        if cfg['temperature'] is None:
            payload.pop('temperature')
        else:
            payload['temperature']=cfg['temperature']
    raw=json.dumps(payload,ensure_ascii=False)
    return hashlib.sha256((tag+raw).encode()).hexdigest(),payload

class Client:
    def __init__(self, path, cfg, transport=None):
        if cfg["model"] not in ("openai/gpt-oss-120b","openai/gpt-6.1-sol","anthropic/claude-sonnet-5.5","openai/gpt-6-astra","anthropic/claude-opus-5.5"):
            raise ValueError("Review the tariff before changing the model")
        minimum={'openai/gpt-oss-120b':(15.86,61),'openai/gpt-6.1-sol':(355.98,1779.92),'anthropic/claude-sonnet-5.5':(355.98,1779.92),'openai/gpt-6-astra':(1779.92,8899.58),'anthropic/claude-opus-5.5':(711.97,3559.84)}[cfg['model']]
        if cfg["input_rub_per_million"]<minimum[0] or cfg["output_rub_per_million"]<minimum[1]:
            raise ValueError("Rates below the verified cost envelope")
        if not 0<cfg["max_output"]<=2048 or cfg["attempt_limit"]<=0 or cfg["rub_limit"]<=0:
            raise ValueError("Invalid experiment limits")
        self.hard_limit = 570 if cfg.get('budget_path') else 480
        if cfg.get('budget_authorization'):
            # A separate frozen authorization is required to extend historical caps.
            # The caller supplies its SHA256; old callers retain the old ceiling.
            from pathlib import Path
            authorization = Path(cfg['budget_authorization']).read_bytes()
            if hashlib.sha256(authorization).hexdigest()!=cfg.get('budget_authorization_sha256'):
                raise ValueError('Budget authorization hash mismatch')
            auth=json.loads(authorization)
            if not cfg.get('budget_path') or auth.get('budget_path')!=str(Path(cfg['budget_path']).resolve()):
                raise ValueError('Budget authorization journal mismatch')
            self.hard_limit=float(auth['cumulative_cap_rub'])
            if not 0<self.hard_limit<=float(auth['source_available_units']):
                raise ValueError('Invalid authorized budget ceiling')
        self.cfg = cfg
        self.db = sqlite3.connect(cfg.get("budget_path",path),timeout=60)
        self.db.execute("CREATE TABLE IF NOT EXISTS calls(id INTEGER PRIMARY KEY, key TEXT, result TEXT)")
        self.shared=cfg.get('budget_path')
        if self.shared:
            self.db.execute('CREATE TABLE IF NOT EXISTS reservations(id INTEGER PRIMARY KEY, model TEXT, key TEXT, charge REAL, result TEXT)')
        self.http = httpx.Client(transport=transport,timeout=cfg.get('timeout',45),trust_env=False)
    def call(self, tag, messages):
        cfg = self.cfg
        key,payload = identity(cfg,tag,messages)
        raw = json.dumps(payload,ensure_ascii=False)
        if len(raw.encode())>cfg.get('max_request_bytes',4000) or cfg["max_output"]>2048:
            raise ValueError("Request outside cost envelope")
        for _ in range(3):
            self.db.execute("BEGIN IMMEDIATE")
            old = self.db.execute("SELECT result FROM calls WHERE key=? AND result IS NOT NULL ORDER BY id DESC",(key,)).fetchall()
            for (entry,) in old:
                saved=json.loads(entry)
                if saved.get("ok"):
                    self.db.commit()
                    return saved
            if cfg.get('strict_dedup'):
                attempts=self.db.execute('SELECT result FROM calls WHERE key=?',(key,)).fetchall()
                if any(r is None for (r,) in attempts):
                    self.db.rollback()
                    raise RuntimeError('Unresolved or in-flight request; no automatic duplicate payment')
                retry_limit=cfg.get('retry_limits_by_key',{}).get(key,cfg.get('max_attempts_per_key',3))
                if not isinstance(retry_limit,int) or not 1<=retry_limit<=6:
                    self.db.rollback()
                    raise ValueError('Invalid bounded per-key recovery allowance')
                if len(attempts)>=retry_limit:
                    self.db.rollback()
                    raise RuntimeError('Per-request lifetime retry limit reached')
            previous=self.db.execute("SELECT result FROM calls").fetchall()
            n=len(previous)
            # Fixed worst-case allowance: 4000 bytes plus 1000 tokens for framing.
            # Reserve the largest cache-write tariff too, before sending. The
            # input bound includes request bytes and framing, not a token guess.
            input_upper=max(cfg["input_rub_per_million"],cfg.get('cache_write_rub_per_million',{'anthropic/claude-sonnet-5.5':711.97,'openai/gpt-6.1-sol':444.98,'openai/gpt-6-astra':2224.9,'anthropic/claude-opus-5.5':1423.94}.get(cfg['model'],0)))
            margin=max(1.05,cfg.get('cost_reserve_multiplier',1.05))
            input_bound=cfg.get('max_input_tokens',5000)
            if cfg.get('bound_request_input'):
                # UTF-8 bytes bound text tokens; 1000 additional tokens cover
                # chat framing. The configured global envelope remains a cap.
                input_bound=min(input_bound,len(raw.encode('utf8'))+1000)
            reserve=(input_bound*input_upper+max(256,cfg["max_output"])*cfg["output_rub_per_million"])/1e6*margin
            exposure=sum(max(reserve, json.loads(v).get('rub',0)) if v else reserve for (v,) in previous)
            if self.shared:
                exposure=self.db.execute('SELECT coalesce(sum(charge),0) FROM reservations').fetchone()[0]
                n=self.db.execute('SELECT count(*) FROM reservations WHERE model=?',(cfg['model'],)).fetchone()[0]
            if n+1>cfg["attempt_limit"] or exposure+reserve>min(cfg["rub_limit"],self.hard_limit):
                self.db.rollback()
                raise RuntimeError("Local spending limit reached")
            attempt=self.db.execute("INSERT INTO calls(key) VALUES (?)",(key,)).lastrowid
            if self.shared:
                reservation=self.db.execute('INSERT INTO reservations(model,key,charge) VALUES (?,?,?)',(cfg['model'],key,reserve)).lastrowid
            self.db.commit()
            start=time.monotonic()
            transient=False
            try:
                response=self.http.post(URL,json=payload,headers={"Authorization":"Bearer "+os.environ["API_KEY"]})
                response.raise_for_status()
                obj=response.json()
                usage=obj["usage"]
                if any(type(usage[k]) is not int or usage[k]<0 for k in ('prompt_tokens','completion_tokens')):
                    raise ValueError('Invalid usage')
                item=obj["choices"][0]
                result=dict(ok=True,answer=item["message"]["content"] or "", finish=item["finish_reason"],
                            model=obj["model"],usage=usage, rub=usage_cost(cfg,usage))
                result['response_id']=obj.get('id')
                result['attempt_id']=attempt
                result['system_fingerprint']=obj.get('system_fingerprint')
                if cfg.get('store_raw_response'):
                    result['raw_response']=obj
                if cfg.get('strict_model') and result['model'] != cfg['model']:
                    result.update(ok=False,error='Model mismatch')
                if usage['prompt_tokens']>input_bound or usage['completion_tokens']>max(256,cfg['max_output']):
                    result.update(ok=False,error='Actual token usage exceeded reserved envelope')
            except httpx.HTTPStatusError as exc:
                status=exc.response.status_code
                result=dict(ok=False,error=f"HTTP {status}")
                transient=status in (429,500,502,503,504)
            except httpx.TransportError as exc:
                result=dict(ok=False,error=type(exc).__name__)
                transient=True
            except (ValueError, KeyError, IndexError, TypeError):
                result=dict(ok=False,error="Malformed response or missing usage")
            result.update(seconds=time.monotonic()-start,started_unix=time.time()-(time.monotonic()-start),request=payload)
            if self.shared:
                # Release unused reserve only after a valid usage-bearing response.
                charge=result['rub']*margin if result.get('ok') else max(reserve,result.get('rub',0)*margin)
                self.db.execute('UPDATE reservations SET charge=?,result=? WHERE id=?',(charge,json.dumps(dict(ok=result.get('ok'),rub=result.get('rub'))),reservation))
            self.db.execute("UPDATE calls SET result=? WHERE id=?",(json.dumps(result,ensure_ascii=False),attempt))
            self.db.commit()
            if not transient:
                return result
            time.sleep(2**_)
        return result

