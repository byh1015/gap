import os, sys, json, re, time, pickle, math, datetime as dt, urllib.request
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np, pandas as pd, yfinance as yf
try:
    import FinanceDataReader as fdr
except Exception:
    fdr = None
import warnings; warnings.filterwarnings("ignore")

KST = dt.timezone(dt.timedelta(hours=9))


def _now():
    """한국 시각. GitHub Actions 러너는 UTC라서 그대로 쓰면 화면 시각과
    뉴스 'N시간 전' 경과분이 9시간 어긋난다."""
    return dt.datetime.now(KST)


HERE = os.environ.get("GAP_HERE") or os.path.dirname(os.path.abspath(__file__))
_HTML_DEF = r"C:\Users\byh10\Documents\gap_today.html"
HTML = os.environ.get("GAP_HTML") or (_HTML_DEF if os.path.isdir(os.path.dirname(_HTML_DEF)) else os.path.join(HERE, "gap_today.html"))
CACHE = os.path.join(HERE, "gap_all_cache.pkl")
NXT_CACHE = os.path.join(HERE, "nxt_list.csv")
NEWS_CACHE = os.path.join(HERE, "gap_news_cache.pkl")
THEME_CACHE = os.path.join(HERE, "theme_cache.pkl")
HIST = os.path.join(HERE, "gap_history.csv")   # 갭 출현 이력(누적)
THEME_EXCL = ("지수", "Index")   # 지수성 테마 제외(넓은 백분류, 그룹 의미 약함)

GAP = 0.05          # 갭 5% 이상
WINDOW = 3          # 최근 3거래일
MIN_VAL = 200       # 갭일 거래대금 하한(억원)
CACHE_HOURS = 6     # 캐시 유효시간
CHUNK = 150         # yfinance 배치 크기

MAT_KW = ("특징주", "핫종목", "실적", "상한가", "급등", "계약", "수주", "공급")
NEWS_KW = ("특징주", "실적", "상한가", "계약", "수주", "공급", "수출", "투자", "인수", "합병",
           "승인", "허가", "임상", "개발", "출시", "양산", "체결", "MOU", "지분", "유상증자")
MAT_DAY_W = 2   # 갭일(당일) 기사 가중치

# 투자자 보호 공시 — 기간/개수 제한 없이 반드시 전부 표시
# (단기과열·투자주의·투자경고는 대개 갭 발생 '이후'에 지정되므로
#  갭일까지만 보던 기존 로직에서는 놓침.  공매도 과열종목도 과열 계열로 포함)
ALERT_KW = (
    ("투자경고", "투자경고"),
    ("투자주의", "투자주의"),
    ("단기과열", "단기과열"),
    ("과열종목", "과열종목"),
)
ALERT_LOOKBACK = 2      # 갭일 이전 N일
ALERT_LOOKAHEAD = 7     # 갭일 이후 N일 (지정은 보통 갭 이후)
NEWS_CACHE_VER = 2      # 공시 구조 변경 시 올려서 캐시 무효화

SCORE_JSON = os.path.join(HERE, "recur_next.json")
SCORE_FALLBACK = {
    "feats": ["gap", "lval", "limup", "bodypos", "kq", "lvm", "rng", "pre5",
              "dist60", "react60", "gapsince", "regime"],
    "coef": [-2.7922, 0.3105, 0.0088, 0.0772, 0.1157, 0.1925, 0.2148, 0.0878,
             -0.0319, 0.0353, 0.2352, -0.3330, 0.0031],
    "mu": [0.0864, 2.9498, 0.0606, 0.4282, 0.6314, 0.4842, 0.0495, 0.0795,
           -0.0857, 1.5636, 193.45, 6.7369],
    "sd": [0.0444, 0.4392, 0.2386, 0.4948, 0.4824, 0.3590, 0.1009, 0.1749,
           0.1190, 1.8821, 351.58, 3.8462],
    "cal": 0.0,
}
W_REC, W_MAT = 1.0, 0.0   # 점수 = 내일 재발확률 단일 기준 (재료는 미검증이라 제외)


def load_score_model():
    try:
        return json.load(open(SCORE_JSON, encoding="utf-8"))
    except Exception:
        return SCORE_FALLBACK


_SM = None


def _feat_row(r):
    vm = r.get("vmult") or 0.0
    return {
        "gap": float(r["g"]),
        "lval": math.log10(max(r.get("val", 1), 1)),
        "limup": 1.0 if r.get("limup") else 0.0,
        "bodypos": 1.0 if r.get("bodypos") else 0.0,
        "kq": 1.0 if r.get("mkt") == "KOSDAQ" else 0.0,
        "lvm": math.log10(vm) if vm > 0 else 0.0,
        "rng": float(r.get("rng", 0.0)),
        "pre5": float(r.get("pre5", 0.0)),
        "dist60": float(r.get("dist60", 0.0)),
        "react60": float(r.get("react60", 0.0)),
        "gapsince": float(r.get("gapsince", 999)),
        "regime": float(r.get("regime", 0.0)),
    }


def rec_prob(r):
    """갭 발생일 기준 내일(다음 거래일) 5%갭 확률(로지스틱 재보정)."""
    global _SM
    if _SM is None:
        _SM = load_score_model()
    fr = _feat_row(r)
    m = _SM
    z = m["coef"][0] + m.get("cal", 0.0)
    for k, f in enumerate(m["feats"]):
        z += m["coef"][k + 1] * ((fr[f] - m["mu"][k]) / m["sd"][k])
    return 1.0 / (1.0 + math.exp(-z))


def apply_scores(rows):
    for r in rows:
        p = rec_prob(r) * 100.0
        mat = float((r.get("mat") or {}).get("score", 0))
        r["rec"] = p
        r["score"] = int(round(W_REC * p + W_MAT * mat))

CARD = """<div class="row{hot}" data-score="{ts}" data-nxt="{nxtf}">
 <div class="l">
  <div class="nm">{name}<span class="cd">{code}</span>{nxt}</div>
  <div class="chips"><span class="chip">{since}</span><span class="chip v">{val}억</span><span class="chip">{mkt}</span>{theme}</div>
  {alerts}
  {news}
 </div>
 <div class="g"><b>{gap}%</b><i class="tr"><em style="width:{w}%"></em></i></div>
 <div class="r"><div class="p">{prc}</div></div>
</div>"""

PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>갭 스캐너 · 전종목 · 최근 {window}거래일 5%+</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#070b14;color:#e8edf8;
font-family:"Pretendard Variable",Pretendard,"Malgun Gothic","맑은 고딕",system-ui,sans-serif;
-webkit-font-smoothing:antialiased;padding-bottom:70px}
.wrap{max-width:1020px;margin:0 auto;padding:30px 20px}
.top{display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:26px}
.brand{font-size:17px;font-weight:800;letter-spacing:-.3px}
.brand i{font-style:normal;color:#ff4d6a}
.brand span{color:#5d6b8c;font-weight:600;margin-left:9px;font-size:13.5px}
.upd{color:#5d6b8c;font-size:12.5px}
.tools{display:flex;align-items:center;gap:12px}
button#filtnxt{font:inherit;font-size:12.5px;font-weight:700;color:#b7c3e6;background:#161e33;border:1px solid #28355e;border-radius:10px;padding:7px 12px;cursor:pointer}button#filtnxt.on{color:#fff;background:#263a7a;border-color:#3a56b8}
button#upd{font:inherit;font-size:13px;font-weight:700;color:#dbe6ff;background:#1a2740;border:1px solid #2c3d63;border-radius:10px;padding:8px 15px;cursor:pointer;transition:.15s ease}
button#upd:hover{background:#22325a;border-color:#3c5488}
button#upd:disabled{opacity:.7;cursor:default}
button#upd.load{color:#ffd9a0;border-color:#5b4a2a;background:#241d10}
button#sort{font:inherit;font-size:13px;font-weight:700;color:#cbd8f5;background:#161f33;border:1px solid #2c3d63;border-radius:10px;padding:8px 13px;cursor:pointer;transition:.15s ease}
button#sort:hover{background:#1d2a45;border-color:#3c5488}
.warn{background:#2a1520;border:1px solid #5b2a3c;color:#ffb3c1;padding:11px 15px;border-radius:11px;margin-bottom:16px;font-size:14px}
.hero h1{font-size:36px;font-weight:800;letter-spacing:-1.2px;margin:0}
.hero h1 em{font-style:normal;color:#ff4d6a}
.hero p{color:#7d8bb0;font-size:14.5px;margin:8px 0 0}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin:26px 0 28px}
.st{background:linear-gradient(155deg,#111a2b,#0b1220);border:1px solid #1d2842;border-radius:16px;padding:17px 18px 15px}
.st b{display:block;font-size:30px;font-weight:800;letter-spacing:-1.2px;font-variant-numeric:tabular-nums;line-height:1.1}
.st span{display:block;color:#6b7aa0;font-size:12.5px;margin-top:5px}
.st.gap b{color:#ff4d6a}
.st.up b{color:#3ddc97}
.rows{display:flex;flex-direction:column;gap:10px}
.row{display:flex;align-items:center;gap:18px;background:#0e1626;border:1px solid #1c2740;border-radius:15px;
padding:16px 20px;transition:.16s ease}
.row:hover{background:#131d33;border-color:#2e3f66;transform:translateY(-1px)}
.row.hot{background:linear-gradient(95deg,#1b1122,#0e1626 65%);border-color:#4a2437}
.l{flex:1;min-width:0}
.nm{font-size:18px;font-weight:700;letter-spacing:-.4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cd{font-size:12.5px;font-weight:600;color:#6b7aa0;background:#161f35;padding:3px 8px;border-radius:7px;margin-left:9px;vertical-align:2px}
.chips{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap}
.chip{font-size:12px;color:#8fa0c4;background:#151e33;border:1px solid #26314f;border-radius:99px;padding:3.5px 10px}
.chip.d{color:#ffd1d9;background:#2b1521;border-color:#4d2436}
.chip.kq{color:#c8b6ff;background:#1e1633;border-color:#3a2c66}
.nxt{font-size:9.5px;font-weight:800;letter-spacing:.1em;color:#4ade80;background:#0d2a1d;border:1px solid #1e4d36;border-radius:4px;padding:1.5px 5px;margin-left:7px;vertical-align:2px}
.grp{display:flex;align-items:center;gap:10px;margin:18px 0 9px;font-size:12.5px;font-weight:800;color:#9fb0cf;letter-spacing:.04em}
.grp::after{content:"";flex:1;height:1px;background:#1f2a44}
.grp .gn{font-weight:600;font-size:11px;color:#65739a}
.grp:first-child{margin-top:4px}
.chip.heat{border-color:transparent}
.chip.heat.h0{color:#8b96ad;background:#212938}
.chip.heat.h1{color:#a9c6ff;background:#233657;border-color:#2f4a76}
.chip.heat.h2{color:#6fe3c6;background:#123f37;border-color:#1e5c4e}
.chip.heat.h3{color:#ffc57a;background:#43311a;border-color:#664a24}
.chip.heat.h4{color:#ff96ab;background:#4a1a26;border-color:#712a3c}
.chip.tot{font-weight:800;letter-spacing:-.2px}
.chip.tot.hi{color:#f0c8ff;background:#3a1f4d;border-color:#6b3f86}
.chip.tot.md{color:#a9d0ff;background:#1f3347;border-color:#3a5f80}
.chip.tot.lo{color:#8b96ad;background:#1d2330;border-color:#333d52}
.chip.thm{color:#7ee0a3;background:#0f2419;border-color:#1e4d36}
.als{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}
.al{display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:800;padding:3.5px 9px;border-radius:7px;text-decoration:none;border:1px solid;line-height:1.55}
.al i{font-style:normal;font-weight:600;opacity:.72;font-variant-numeric:tabular-nums;letter-spacing:0}
.al:hover{filter:brightness(1.3);text-decoration:none}
.al-w{color:#ffc9c9;background:#3a1418;border-color:#7d2b32}
.al-c{color:#ffdfad;background:#3a2a12;border-color:#7d5a24}
.al-h{color:#cfe6ff;background:#13283c;border-color:#2a5b86}
.g{width:132px;text-align:right}
.g b{display:block;font-size:27px;font-weight:800;color:#ff8a3d;letter-spacing:-1.4px;font-variant-numeric:tabular-nums}
.tr{display:block;height:5px;border-radius:5px;background:#1c2740;margin-top:8px;overflow:hidden}
.tr em{display:block;height:100%;border-radius:5px;background:linear-gradient(90deg,#ff4d6a,#ff9a3d)}
.r{width:128px;text-align:right}
.p{font-size:17.5px;font-weight:700;letter-spacing:-.5px;font-variant-numeric:tabular-nums}
.a{font-size:13.5px;font-weight:700;margin-top:5px;font-variant-numeric:tabular-nums}
.a.up{color:#3ddc97}.a.dn{color:#ff6b8a}
.nws{margin-top:10px}
.nws summary{list-style:none;cursor:pointer;display:inline-flex;align-items:center;gap:6px;font-size:12.5px;color:#93a4c8;user-select:none}
.nws summary::-webkit-details-marker{display:none}
.nws summary:hover{color:#cdd9f2}
.nws summary .nd{color:#4f5f82;font-weight:600}
.nws summary .nsum{color:#6b7794;font-variant-numeric:tabular-nums}
.nws summary::after{content:"▾";font-size:10px;color:#4f5f82;transition:transform .15s}
.nws[open] summary::after{transform:rotate(180deg)}
.nws .nlist{display:flex;flex-direction:column;gap:3px;margin-top:6px;padding-left:2px}
.nws .nlist a{font-size:12.5px;color:#93a4c8;text-decoration:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%}
.nws .nlist a:hover{color:#cdd9f2;text-decoration:underline}
.nws .nlist a.fresh{color:#8ff0b5}.nws .nlist a.fresh .nd{color:#3ddc97}
.nws .nlist a.warm{color:#d7c98a}.nws .nlist a.warm .nd{color:#b9a54e}
.nws .nlist a.stale{color:#6b7794}.nws .nlist a.stale .nd{color:#4f5f82}
.nws .nd{color:#4f5f82;font-variant-numeric:tabular-nums;margin-right:7px;font-weight:600}
.nws .no{font-size:12.5px;color:#4f5f82}
.empty{color:#6b7aa0;text-align:center;padding:54px 10px;background:#0e1626;border:1px dashed #26314f;border-radius:15px;font-size:16px}
.foot{color:#5d6b8c;font-size:13px;line-height:1.9;margin-top:26px}
@media(max-width:660px){
 .wrap{padding:18px 13px}
 .top{gap:10px;margin-bottom:18px}
 .brand{font-size:15px}.brand span{font-size:12px;margin-left:6px}
 .tools{gap:8px;width:100%}
 .upd{font-size:11.5px;order:3;width:100%}
 button#upd,button#sort{font-size:12px;padding:7px 11px}
 .hero h1{font-size:25px}
 .hero p{font-size:13px}
 .stats{grid-template-columns:repeat(2,1fr);gap:9px;margin:18px 0 20px}
 .st{padding:13px 14px 11px;border-radius:13px}
 .st b{font-size:24px}
 .st span{font-size:11.5px}
 .rows{gap:9px}
 .row{flex-wrap:wrap;padding:13px 14px;gap:8px 12px}
 .l{flex:1 1 100%;order:1}
 .nm{font-size:16px}
 .cd{display:none}
 .chips{margin-top:6px;gap:5px}
 .chip{font-size:11px;padding:3px 8px}
 .g{width:auto;order:2;flex:0 0 auto;text-align:left}
 .g b{font-size:23px;display:inline-block}
 .tr{display:none}
 .r{width:auto;order:3;flex:1;text-align:right}
 .p{font-size:15.5px}
 .a{font-size:12.5px;margin-top:3px}
 .nws summary{font-size:12px}
 .nws .nlist a{font-size:12px;white-space:normal;line-height:1.45}
 .foot{font-size:12px;line-height:1.75;margin-top:20px}
}
</style></head>
<body><div class="wrap">
 <div class="top">
  <div class="brand">GAP<i>&middot;</i>SCAN <span>전종목 스캐너</span></div>
   <div class="tools"><span class="upd">데이터 {when} (KST)</span><button id="filtnxt" type="button">NXT만</button><button id="sort" type="button">&#8645; 내일순</button><button id="upd" type="button">&#8635; 업데이트</button></div> </div>
 {warn}
 <div class="hero">
  <h1>갭 <em>5%</em> 이상</h1>
  <p>전종목 KOSPI+KOSDAQ &middot; 최근 {window}거래일 &middot; 시가 &gt; 전일종가 &times; 1.05 &middot; 갭일 거래대금 {minval}억+</p>
 </div>
 <div class="stats">
  <div class="st gap"><b>{total}</b><span>감지 종목</span></div>
  <div class="st"><b>{today}</b><span>최근 1~2일 갭</span></div>
  <div class="st"><b>{maxgap}</b><span>최고 갭</span></div>
  <div class="st up"><b>{nxt}</b><span>NXT 거래가능</span></div>
 </div>
 <div class="rows">
{rows}
 </div>
 <div class="foot">
  갭 = 시가 &divide; 전일종가 &minus; 1 &nbsp;&middot;&nbsp; &lsquo;갭 이후&rsquo;는 갭 발생일 종가 대비 현재 등락 &nbsp;&middot;&nbsp; 주황 바는 15% 기준<br>
   다음 거래일 시가가 전일종가 대비 5% 이상으로 갭업할 확률. &nbsp;&middot;&nbsp; 같은 테마가 함께 있는 종목만 `동일테마` 배지 &nbsp;&middot;&nbsp; 우상단 &#8645; 로 내일순 정렬<br>
  스캔 {univ}종목 (데이터 확보) &middot; 우상단 &#8635; 업데이트 &rarr; GitHub Actions 에서 Run workflow (약 2분)
 </div>
</div>
<script>
(function(){
  var fn=document.getElementById('filtnxt');
  if(fn){var on=false;fn.addEventListener('click',function(){on=!on;fn.classList.toggle('on',on);document.querySelectorAll('.row').forEach(function(r){if(!on){r.style.display='';return;}r.style.display=r.getAttribute('data-nxt')==='1'?'':'none';});});}

 var b=document.getElementById('upd'); if(!b) return;
 var host=location.hostname;
 var pages=host.indexOf('github.io')>=0;
 var orig='\u21bb \uc5c5\ub370\uc774\ud2b8';
 function set(t,d){b.textContent=t;b.disabled=d;b.classList.toggle('load',d);}
 b.addEventListener('click',function(){
   if(pages){
     b.textContent='\uac31\uc2e0 \ud398\uc774\uc9c0\u2026';
     location.href='https://github.com/byh1015/gap/actions/workflows/gap-scan.yml';
     return;
   }
   if(location.protocol!=='http:'&&location.protocol!=='https:'){alert('\uc11c\ubc84 \ubaa8\ub4dc\ub85c \uc5f4\uc5b4\uc57c \ud569\ub2c8\ub2e4.');return;}
   set('\uac31\uc2e0 \uc911\u2026',true);
   fetch('/update',{method:'POST'}).then(function(r){return r.json();}).then(function(){
     var iv=setInterval(function(){
       fetch('/status').then(function(r){return r.json();}).then(function(s){
         if(!s.updating){clearInterval(iv);
           if(s.error){set(orig,false);alert('\uac31\uc2e0 \uc2e4\ud328: '+s.error);}
           else{set('\uc644\ub8cc \u00b7 \ub2e4\uc2dc \uc5ec\ub294 \uc911',true);location.reload();}
         }
       }).catch(function(){});
     },2000);
   }).catch(function(){set(orig,false);alert('\uc11c\ubc84 \uc751\ub2f5 \uc5c6\uc74c');});
 });
})();
 (function(){
  var sb=document.getElementById('sort'); if(!sb) return;
  var wrap=document.querySelector('.rows'); if(!wrap) return;
  var orig=Array.prototype.slice.call(wrap.children);
  var cards=Array.prototype.slice.call(wrap.querySelectorAll('.row'));
  var grps=Array.prototype.slice.call(wrap.querySelectorAll('.grp'));
  var byScore=false;
  function saveOpen(){
   return cards.map(function(c){var d=c.querySelector('details.nws');return !!(d&&d.open);});
  }
  function restoreOpen(st){
   cards.forEach(function(c,i){var d=c.querySelector('details.nws');if(d)d.open=!!st[i];});
  }
  sb.addEventListener('click',function(){
   var st=saveOpen();
   if(!byScore){
     cards.slice().sort(function(a,b){return (parseFloat(b.getAttribute('data-score'))||0)-(parseFloat(a.getAttribute('data-score'))||0);}).forEach(function(el){wrap.appendChild(el);});
     grps.forEach(function(g){g.style.display='none';});
     sb.innerHTML='&#8645; \ucd5c\uc2e0\uc21c'; byScore=true;
   } else {
     orig.forEach(function(el){wrap.appendChild(el);});
     grps.forEach(function(g){g.style.display='';});
     sb.innerHTML='&#8645; \ub0b4\uc77c\uc21c'; byScore=false;
   }
   restoreOpen(st);
  });
 })();
</script>
</body></html>"""


def load_nxt():
    try:
        df = pd.read_csv(NXT_CACHE, dtype={"code": str})
        return set(df["code"].str.zfill(6))
    except Exception:
        return set()


def _news_fetch(code, until, max_pages=8):
    """최신순 페이징. 게시일이 until 이하가 되면 중단.
    -> [(date, title, body, id, total, url, dt)]"""
    out = []
    for pg in range(1, max_pages + 1):
        u = "https://m.stock.naver.com/api/news/stock/%s?pageSize=25&page=%d" % (code, pg)
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            d = json.loads(urllib.request.urlopen(req, timeout=12).read().decode("utf-8"))
        except Exception:
            break
        if not d:
            break
        oldest = None
        for e in d:
            tot = int(e.get("total", 1) or 1)
            for it in (e.get("items") or []):
                ds = it.get("datetime", "")
                if len(ds) < 8:
                    continue
                try:
                    dd = dt.date(int(ds[:4]), int(ds[4:6]), int(ds[6:8]))
                except Exception:
                    continue
                oldest = dd if oldest is None else min(oldest, dd)
                out.append((dd, it.get("title", "").strip(), it.get("body", ""),
                            it.get("id", ""), tot,
                            it.get("mobileNewsUrl") or it.get("titleFull") or "", ds))
        if oldest is None or oldest <= until:
            break
    return out


def news_for(gap_date, items):
    """갭 발생일 전후 기사 중 최대 2건 -> [(mm/dd, title, url)]"""
    gd = dt.date.fromisoformat(gap_date)
    near, before = [], []
    for rec in items:
        dd, ti, body, iid, tot, url = rec[:6]
        gap = (dd - gd).days
        mmdd = "%02d/%02d" % (dd.month, dd.day)
        if -1 <= gap <= 1:
            near.append((mmdd, ti, url))
        elif gap < -1:
            before.append((mmdd, ti, url))
    pool = near if near else before[:2]
    if not pool:
        return None
    return pool[:2]


NEWS_NOISE = ("주식마감", "[마감]", "장마감", "마감시황", "증시]", "[증시",
              "거래대금", "종목NOW", "장분석", "장초반", "[개장", "[장중",
              "장 분석", "급등 종목", "상한가 종목", "특징주 정리", "외국인", "기관 순매",
              "거래 1위", "거래1위", "거래상위", "거래 상위", "거래 上", "상위 종목",
              "대장주", "반도체 대표주", "종목별", "업종별", "테마주 약세", "테마 약세",
              "애프터마켓", "후장", "마켓리뷰", "장마감 리뷰", "시간외")


def _news_related(name, title):
    for w in NEWS_NOISE:
        if w in title:
            return False
    if name and name in title:
        return True
    if name and len(name) >= 5 and name[:5] in title:
        return True
    for w in NEWS_KW:
        if w in title:
            return True
    return False



def related_news(code, gap_date, name="", keys=(), pages=3):
    """갭 발생 전날 ~ 현재까지의 '관련' 기사 목록(최신순).
    종목명이 제목에 직접 있으면 우선 채택. 종목명 기사가 부족할 때만
    재료 키워드(NEWS_KW) 기사로 보충(최대 3건).
    네이버 종목뉴스 + 네이버 뉴스검색(사회이슈 포함)을 합쳐 중복 제거.
    -> [(경과분, 제목, url, 발행일)] 최신순 (최대 8건). 경과분 None=시각미상."""
    lo = dt.date.fromisoformat(gap_date) - dt.timedelta(days=1)
    cut = _now().date() - dt.timedelta(days=60)   # 검색 커버 한계
    lo = max(lo, cut)
    lo_min = (_now().date() - lo).days * 1440
    now = _now()

    def named(ti):
        return bool(name) and (name in ti or (len(name) >= 5 and name[:5] in ti))

    items = _news_fetch(code, lo - dt.timedelta(days=14), max_pages=pages)
    seen = set()
    strong, weak = [], []

    def put(mins, ti, url, dd):
        if not _news_related(name, ti):
            return
        k = re.sub(r"[^0-9a-z\uac00-\ud7a3]", "", ti.lower())
        if not k or k in seen:
            return
        seen.add(k)
        (strong if named(ti) else weak).append((mins, ti, url, dd))

    for x in items:
        dd, ti, body, iid, tot, url, ds = x
        if dd < lo or dd > _now().date():
            continue
        mins = None
        if len(ds) >= 12:
            try:
                d2 = dt.datetime(int(ds[:4]), int(ds[4:6]), int(ds[6:8]),
                                 int(ds[8:10]), int(ds[10:12]))
                mins = int((now - d2).total_seconds() // 60)
            except Exception:
                mins = None
        if mins is None:
            mins = (_now().date() - dd).days * 1440
        put(mins, ti, url, dd)
    if name:
        for age, ti, url in naver_search_news(name, limit=20):
            if age is not None and age > lo_min:
                continue
            dd = _now().date() - dt.timedelta(days=(age or 0) // 1440)
            put(age, ti, url, dd)
    key = lambda x: (x[0] if x[0] is not None else 10 ** 9)
    strong.sort(key=key)
    weak.sort(key=key)
    out = strong[:8]
    if len(out) < 4:
        out += weak[:3 if not out else 2]
    out.sort(key=key)
    return out[:8]








def _theme_keys(theme_names):
    """테마명 -> 매칭용 키워드(괄호/슬래시 분해 후 2자 이상)."""
    out = set()
    for t in theme_names:
        for part in re.split(r"[()\[\]/·,·]", t):
            p = part.strip()
            if len(p) >= 2 and p not in ("등", "관련", "대표주", "테마"):
                out.add(p)
    return out


_NS_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
          "Referer": "https://search.naver.com/"}


def _parse_age(txt):
    """'3시간 전' / '2일 전' / '2026.10.19.' -> 경과일(int) 또는 None."""
    if not txt:
        return None
    m = re.search(r"(\d+)\s*분\s*전", txt)
    if m:
        return 0
    m = re.search(r"(\d+)\s*시간\s*전", txt)
    if m:
        return 0
    m = re.search(r"(\d+)\s*일\s*전", txt)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d{4})\.(\d{1,2})\.(\d{1,2})", txt)
    if m:
        try:
            d = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return (_now().date() - d).days
        except Exception:
            return None
    return None


def _parse_min(txt):
    """기사 시각 텍스트 -> 분 단위 경과(int) 또는 None."""
    if not txt:
        return None
    m = re.search(r"(\d+)\s*분\s*전", txt)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*시간\s*전", txt)
    if m:
        return int(m.group(1)) * 60
    m = re.search(r"(\d+)\s*일\s*전", txt)
    if m:
        return int(m.group(1)) * 1440
    m = re.search(r"(\d{4})\.(\d{1,2})\.(\d{1,2})", txt)
    if m:
        try:
            d = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return (_now().date() - d).days * 1440
        except Exception:
            return None
    return None



def naver_search_news(query, sort=1, limit=30):
    """네이버 뉴스검색 -> [(경과분, 제목, url)] 최신순.
    종목명으로 검색하면 사회이슈·정치 등 가격 무관 기사도 포함된다."""
    u = "https://search.naver.com/search.naver?where=news&query=%s&sort=%d" % (
        urllib.parse.quote(query), sort)
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers=_NS_UA), timeout=15).read().decode("utf-8", "replace")
    except Exception:
        return []
    # 헤드라인 앵커 위치와 직전 날짜 텍스트를 짝지어 파싱
    heads = list(re.finditer(
        r'<a[^>]+href="(https?://[^"]+)"[^>]*>\s*<span[^>]*sds-comps-text-type-headline[^>]*>([^<]+)</span>', r))
    dates = list(re.finditer(r'(\d+\s*분\s*전|\d+\s*시간\s*전|\d+\s*일\s*전)', r))
    dd = [(_parse_min(m.group(1)), m.start()) for m in dates]
    out = []
    for hm in heads:
        ti = re.sub("<[^>]+>", "", hm.group(2)).strip()
        for a, b in (("&quot;", '"'), ("&#39;", "'"), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">")):
            ti = ti.replace(a, b)
        if not ti:
            continue
        # 이 헤드라인 바로 앞(2.5KB 이내)의 가장 가까운 날짜
        age = None
        best = -1
        for a, pos in dd:
            if pos < hm.start() and (hm.start() - pos) <= 2500 and pos > best:
                best = pos
                age = a
        out.append((age, ti, hm.group(1)))
        if len(out) >= limit:
            break
    return out




def _keep(name, title):
    if name and name in title:
        return True
    if name and len(name) >= 5 and name[:5] in title:
        return True
    for w in MAT_KW:
        if w in title:
            return True
    return False


def _tier(s):
    if s >= 80:
        return 4
    if s >= 55:
        return 3
    if s >= 30:
        return 2
    if s >= 10:
        return 1
    return 0


TIER_NAME = ("없음", "약", "보통", "강", "폭발")
MAT_LO, MAT_HI = 2, 1   # 재료 창: [갭-LO, 갭+HI]


def material_score(gap_date, name, items):
    """뉴스 기반 재료 트래픽: N=관련기사수, S=Σ확산(total)."""
    gd = dt.date.fromisoformat(gap_date)
    lo, hi = gd - dt.timedelta(days=MAT_LO), gd + dt.timedelta(days=MAT_HI)
    seen = {}
    for dd, ti, body, iid, tot, url, ds in items:
        if lo <= dd <= hi and _keep(name, ti):
            k = re.sub(r"[^0-9a-z\uac00-\ud7a3]", "", ti.lower())
            if k and k not in seen:
                seen[k] = (dd == gd, tot)
    n = len(seen)
    s = sum(t for _, t in seen.values())
    nd = sum(1 for day, _ in seen.values() if day)
    sd = sum(t for day, t in seen.values() if day)
    W = MAT_DAY_W
    N = n + (W - 1) * nd
    S = s + (W - 1) * sd
    sc = min(100, round(16.1 * math.log(1 + N) + 14.8 * math.log(1 + S))) if n else 0
    return {"n": n, "s": s, "nd": nd, "score": sc, "tier": _tier(sc)}


def _fetch_disc_raw(code):
    u = "https://m.stock.naver.com/api/stock/%s/disclosure" % code
    req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
    d = json.loads(urllib.request.urlopen(req, timeout=12).read().decode("utf-8"))
    out = []
    for it in d:
        out.append((it.get("datetime", ""), it.get("title", "").strip()))
    return out


def alert_label(title):
    """제목에서 투자자 보호 공시 종류 라벨. 없으면 None.
    ALERT_KW 순서 = 심각도 순(투자경고 > 투자주의 > 단기과열 > 과열종목)."""
    for kw, lab in ALERT_KW:
        if kw in title:
            return lab
    return None


def disclosure_for(code, gap_date):
    """갭일 기준 공시.

    투자자 보호 공시(투자경고/투자주의/단기과열/공매도 과열)는 대개 갭 발생
    '이후'에 지정되므로 기간·개수 제한 없이 전부 채택한다.
    그 외 공시는 갭일 이전 2일~당일, 최대 2건(기존과 동일).

    -> {"alert": [(mmdd, title, url, kind)...], "normal": [(mmdd, title, url)...]}
    """
    try:
        items = _fetch_disc_raw(code)
    except Exception:
        return {"alert": [], "normal": []}
    gd = dt.date.fromisoformat(gap_date)
    url = "https://m.stock.naver.com/domestic/stock/%s/management/disclosure" % code
    alerts, normals = [], []
    for ds, ti in items:
        if not ds:
            continue
        try:
            d = dt.datetime.fromisoformat(ds).date()
        except Exception:
            continue
        mmdd = ds[5:7] + "/" + ds[8:10]
        lag = (gd - d).days          # 양수 = 갭일 이전, 음수 = 갭일 이후
        kind = alert_label(ti)
        if kind is not None:
            if -ALERT_LOOKAHEAD <= lag <= ALERT_LOOKBACK:
                alerts.append((mmdd, ti, url, kind))
        elif 0 <= lag <= 2:
            normals.append((mmdd, ti, url))
    alerts.sort(key=lambda x: [k for _, k in ALERT_KW].index(x[3]))
    return {"alert": alerts[:8], "normal": normals[:2]}


def _api_json(u):
    req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
    return json.loads(urllib.request.urlopen(req, timeout=15).read().decode("utf-8"))


def get_themes(force=False):
    """{"m": 코드->[테마명...], "size": 테마명->종목수} (Naver 테마). 24h 캐시."""
    if (not force) and os.path.exists(THEME_CACHE) and (time.time() - os.path.getmtime(THEME_CACHE)) < 24 * 3600:
        try:
            c = pickle.load(open(THEME_CACHE, "rb"))
            if isinstance(c, dict) and "m" in c and "size" in c:
                return c
        except Exception:
            pass
    groups = []
    size = {}
    try:
        pg = 1
        while pg <= 20:
            d = _api_json("https://m.stock.naver.com/api/stocks/theme?page=%d&pageSize=100" % pg)
            gs = d.get("groups") or []
            for g in gs:
                groups.append((g["no"], g["name"]))
                size[g["name"]] = int(g.get("totalCount", 0) or 0)
            if not gs or len(groups) >= int(d.get("totalCount", 0)):
                break
            pg += 1
    except Exception:
        pass
    m = {}
    for no, name in groups:
        pg = 1
        try:
            while pg <= 30:
                d = _api_json("https://m.stock.naver.com/api/stocks/theme/%d?page=%d&pageSize=100" % (no, pg))
                ss = d.get("stocks") or []
                for s in ss:
                    m.setdefault(s["itemCode"], []).append(name)
                psz = int(d.get("pageSize", 20) or 20)
                tot = int(d.get("totalCount", 0) or 0)
                if not ss or pg * psz >= tot:
                    break
                pg += 1
        except Exception:
            pass
        time.sleep(0.02)
    out = {"m": m, "size": size}
    if m:
        pickle.dump(out, open(THEME_CACHE, "wb"))
    return out


def get_news(rows, force=False):
    cache = {}
    if (not force) and os.path.exists(NEWS_CACHE) and (time.time() - os.path.getmtime(NEWS_CACHE)) < CACHE_HOURS * 3600:
        try:
            cache = pickle.load(open(NEWS_CACHE, "rb"))
        except Exception:
            cache = {}
    changed = False
    for r in rows:
        k = (r["code"], r["date"])
        c = cache.get(k) or {}
        if force or k not in cache or "mat" not in c or c.get("ver") != NEWS_CACHE_VER:
            until = dt.date.fromisoformat(r["date"]) - dt.timedelta(days=MAT_LO)
            try:
                items = _news_fetch(r["code"], until)
            except Exception:
                items = []
            try:
                disc = disclosure_for(r["code"], r["date"])
            except Exception:
                disc = {"alert": [], "normal": []}
            cache[k] = {"news": news_for(r["date"], items),
                        "disc": disc,
                        "ver": NEWS_CACHE_VER,
                        "mat": material_score(r["date"], r.get("name", ""), items)}
            changed = True
    if changed:
        try:
            pickle.dump(cache, open(NEWS_CACHE, "wb"))
        except Exception:
            pass
    for r in rows:
        e = cache.get((r["code"], r["date"])) or {}
        r["news"] = e.get("news")
        d = e.get("disc") or {"alert": [], "normal": []}
        if isinstance(d, list):          # 옛 캐시 호환
            d = {"alert": [], "normal": d}
        r["disc"] = d
        r["mat"] = e.get("mat") or {"n": 0, "s": 0, "nd": 0, "score": 0, "tier": 0}
        try:
            r["latest"] = related_news(r["code"], r["date"], r.get("name", ""),
                                       _theme_keys(r.get("themes") or []))
        except Exception:
            r["latest"] = None


def get_universe():
    """전종목 코드->이름. universe.csv 우선(오프라인/CI용), 없으면 FDR."""
    codes = {}
    up = os.path.join(HERE, "universe.csv")
    if os.path.exists(up):
        import csv as _csv
        with open(up, "r", encoding="utf-8-sig", newline="") as f:
            for row in _csv.DictReader(f):
                t = (row.get("ticker") or "").strip()
                n = (row.get("name") or "").strip()
                if t:
                    codes[t] = n
        if codes:
            return codes
    if fdr is None:
        raise RuntimeError("universe.csv 없음이고 FinanceDataReader도 미설치")
    for mk, suf in [("KOSPI", ".KS"), ("KOSDAQ", ".KQ")]:
        for _, r in fdr.StockListing(mk).iterrows():
            codes[str(r["Code"]) + suf] = str(r["Name"])
    return codes


def download_all(tk):
    frames = {}
    nb = (len(tk) + CHUNK - 1) // CHUNK
    for i in range(0, len(tk), CHUNK):
        batch = tk[i:i + CHUNK]
        try:
            d = yf.download(batch, period="1mo", interval="1d", progress=False,
                            auto_adjust=False, group_by="ticker", threads=True)
            lvl0 = set(d.columns.get_level_values(0)) if len(d) else set()
            for t in batch:
                if t in lvl0:
                    df = d[t].dropna()
                    if len(df) >= 5: frames[t] = df
        except Exception:
            pass
        sys.stdout.write("\r다운로드 %d/%d  확보 %d" % (i // CHUNK + 1, nb, len(frames)))
        sys.stdout.flush()
        time.sleep(2.5)
    print()
    return frames


def get_data(force=False):
    if (not force) and os.path.exists(CACHE) and (time.time() - os.path.getmtime(CACHE)) < CACHE_HOURS * 3600:
        try:
            with open(CACHE, "rb") as f:
                return pickle.load(f), True
        except Exception:
            pass
    codes = get_universe()
    frames = download_all(list(codes))
    obj = {"frames": frames, "codes": codes,
           "when": _now().strftime("%Y-%m-%d %H:%M")}
    with open(CACHE, "wb") as f:
        pickle.dump(obj, f)
    return obj, False


def scan(obj):
    frames, codes = obj["frames"], obj["codes"]
    nxt = load_nxt()
    daily = {}
    for t, d in frames.items():
        o = d["Open"].values; c = d["Close"].values; idx = d.index
        for i in range(1, len(c)):
            if c[i - 1] > 0 and o[i] / c[i - 1] - 1 >= GAP:
                k = idx[i].date(); daily[k] = daily.get(k, 0) + 1
    if daily:
        s = pd.Series(daily); s.index = pd.to_datetime(s.index); s = s.sort_index()
        reg = s.rolling(20, min_periods=3).mean()
    else:
        reg = pd.Series(dtype=float)
    rows = []
    for t, d in frames.items():
        o = d["Open"].values; c = d["Close"].values; v = d["Volume"].values
        hi = d["High"].values; low = d["Low"].values; idx = d.index; n = len(c)
        if n < 2: continue
        dollar = (v * c) / 1e8
        volma = pd.Series(v).rolling(20).mean().values
        gapidx = [i for i in range(1, n) if c[i - 1] > 0 and o[i] / c[i - 1] - 1 >= GAP]
        lo = max(1, n - WINDOW)
        code = t.split(".")[0]
        mkt = "KOSDAQ" if t.endswith(".KQ") else "KOSPI"
        for i in range(lo, n):
            g = o[i] / c[i - 1] - 1
            if g < GAP: continue
            val = float(dollar[i])
            if val < MIN_VAL: continue
            vmult = float(v[i] / volma[i]) if (not np.isnan(volma[i]) and volma[i] > 0) else 1.0
            limup = bool(i >= 2 and (c[i - 1] / c[i - 2] - 1) >= 0.29)
            prev = [k for k in gapidx if k < i]
            react60 = sum(1 for k in prev if k >= i - 60)
            gapsince = (i - prev[-1]) if prev else 999
            hh = c[max(0, i - 60):i + 1].max()
            rng = (hi[i] - low[i]) / c[i]
            pre5 = (c[i - 1] / c[i - 6] - 1) if i >= 6 else 0.0
            dist60 = (c[i] / hh - 1) if hh > 0 else 0.0
            regv = reg.get(pd.Timestamp(idx[i].date()), 0.0)
            rows.append(dict(
                name=codes.get(t, t), code=code, nxt=(code in nxt), mkt=mkt,
                date=str(idx[i].date()), g=float(g), since=n - 1 - i, val=val,
                close=float(c[-1]), after=float(c[-1] / c[i] - 1),
                bodypos=bool(c[i] > o[i]), limup=limup, vmult=vmult,
                rng=float(rng), pre5=float(pre5), dist60=float(dist60),
                react60=react60, gapsince=min(gapsince, 999),
                regime=(float(regv) if np.isfinite(regv) else 0.0)))
    best = {}
    for r in rows:
        k = r["code"]
        if k not in best or r["date"] > best[k]["date"]: best[k] = r
    rows = sorted(best.values(), key=lambda r: (r["date"], r["g"]), reverse=True)
    return rows, len(frames)


HIST_COLS = ["date", "code", "name", "mkt", "g", "open", "prev_close",
             "close", "high", "low", "ret_day", "val", "next_date",
             "next_open", "next_close", "ret_next", "ret_next_open"]


def record_history(obj):
    """갭 출현 이력을 gap_history.csv 에 누적한다.

    프레임 윈도우(약 1mo) 안의 갭은 매번 다시 계산해 가장 완전한 값으로 채우고,
    그 창 밖으로 벗어난 옛 행은 그대로 남겨 두므로 기록은 무한히 쌓인다.
    MIN_VAL 필터는 걸지 않는다 — 나중에 거래대금 임계값을 되짚어 최적화할 수
    있도록 갭 5% 이상 전량을 기록한다.  예외가 나도 스캔 결과는 내보낸다."""
    try:
        old = pd.read_csv(HIST, dtype=str)
    except Exception:
        old = None
    keep = []
    seen = set()
    frames, codes = obj["frames"], obj["codes"]
    for tk, d in frames.items():
        code = tk.split(".")[0]
        name = codes.get(tk, tk)
        mkt = "KOSDAQ" if tk.endswith(".KQ") else "KOSPI"
        o = d["Open"].values; hi = d["High"].values
        lw = d["Low"].values; c = d["Close"].values; v = d["Volume"].values
        idx = d.index; n = len(c)
        for i in range(1, n):
            if c[i - 1] <= 0:
                continue
            g = o[i] / c[i - 1] - 1
            if g < GAP:
                continue
            gd = str(idx[i].date())
            seen.add((gd, code))
            opn = float(o[i]); cl = float(c[i])
            row = {"date": gd, "code": code, "name": name, "mkt": mkt,
                   "g": round(float(g), 5), "open": round(opn, 2),
                   "prev_close": round(float(c[i - 1]), 2), "close": round(cl, 2),
                   "high": round(float(hi[i]), 2), "low": round(float(lw[i]), 2),
                   "ret_day": (round(cl / opn - 1, 5) if opn > 0 else ""),
                   "val": round(float(v[i] * c[i] / 1e8), 1),
                   "next_date": "", "next_open": "", "next_close": "",
                   "ret_next": "", "ret_next_open": ""}
            if i + 1 < n:
                no = float(o[i + 1]); nc = float(c[i + 1])
                row["next_date"] = str(idx[i + 1].date())
                row["next_open"] = round(no, 2)
                row["next_close"] = round(nc, 2)
                if cl > 0:
                    row["ret_next"] = round(nc / cl - 1, 5)
                    if no > 0:
                        row["ret_next_open"] = round(nc / no - 1, 5)
            keep.append(row)
    n_old = 0
    if old is not None and len(old):
        for r in old.to_dict("records"):
            k = (r.get("date"), str(r.get("code", "")).zfill(6))
            if k[0] and k[1] and k not in seen:
                r["code"] = k[1]
                keep.append({c0: r.get(c0, "") for c0 in HIST_COLS})
                n_old += 1
    if not keep:
        return {"rows": 0}
    df = pd.DataFrame(keep, columns=HIST_COLS)
    df = df.sort_values(["date", "g"], ascending=[False, False], kind="mergesort")
    tmp = HIST + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, HIST)
    return {"rows": len(df), "new": len(keep) - n_old, "kept": n_old}


def _age_label(mins):
    """경과분 -> (라벨, 색클래스). 24시간 내는 'N시간 전'/'N분 전'."""
    if mins is None:
        return "오늘", "fresh"
    if mins < 60:
        return "%d분 전" % max(1, mins), "fresh"
    if mins < 1440:
        return "%d시간 전" % (mins // 60), "fresh"
    days = mins // 1440
    if days == 1:
        return "어제", "fresh"
    if days <= 2:
        return "%d일 전" % days, "fresh"
    if days <= 7:
        return "%d일 전" % days, "warm"
    return "%d일 전" % days, "stale"


def render(rows, nuniv, when, failed=False):
    import html as _h
    esc = _h.escape
    n_nxt = sum(1 for r in rows if r["nxt"])
    today = sum(1 for r in rows if r["since"] <= 1)
    cards = []
    for r in rows:
        gap = r["g"] * 100
        since = "오늘" if r["since"] == 0 else "%d일 전" % r["since"]
        nws = r.get("news")
        latest = r.get("latest") or []
        news_html = ""
        if latest:
            rows_html = []
            for it in latest:
                mins, ti, u, _d = (list(it) + [None, None])[:4]
                lab, cls = _age_label(mins)
                rows_html.append('<a class="%s" href="%s" target="_blank" rel="noopener"><span class="nd">%s</span>%s</a>'
                                 % (cls, esc(u or "#"), lab, esc(ti)))
            lab0, _c0 = _age_label(latest[0][0])
            news_html = ('<details class="nws"><summary><span class="nd">뉴스</span>'
                         '<span class="nsum">%s · %d건</span></summary>'
                         '<div class="nlist">%s</div></details>'
                         % (esc(lab0), len(latest), "".join(rows_html)))
        sh = r.get("shared") or []
        if sh:
            parts = ["%s(%d)" % (esc(t), c) for t, c in sh[:2]]
            theme_html = '<span class="chip thm" title="현재 목록에서 같은 테마">%s</span>' % " · ".join(parts)
        else:
            theme_html = ""
        # 투자자 보호 공시(투자경고/투자주의/단기과열/공매도 과열) — 접히지 않고 항상 표시
        alerts_html = ""
        al = (r.get("disc") or {}).get("alert") or []
        if al:
            cls = {"투자경고": "w", "투자주의": "c", "단기과열": "h", "과열종목": "h"}
            seen, boxes = set(), []
            for mmdd, ti, u, kind in al:
                key = (kind, mmdd)
                if key in seen:
                    continue
                seen.add(key)
                boxes.append('<a class="al al-%s" href="%s" target="_blank" rel="noopener" title="%s">'
                             '<b>%s</b><i>%s</i></a>'
                             % (cls.get(kind, "h"), esc(u or "#"), esc(ti), esc(kind), esc(mmdd)))
            if boxes:
                alerts_html = '<div class="als">%s</div>' % "".join(boxes)
        ts = int(r.get("score", 0))
        tcls = "hi" if ts >= 40 else ("md" if ts >= 30 else "lo")
        rp = "%d" % round(r.get("rec", 0.0))
        cards.append(CARD
            .replace("{hot}", " hot" if gap >= 7 else "")
            .replace("{ts}", str(ts))
            .replace("{tcls}", tcls)
            .replace("{rp}", rp)
            .replace("{name}", esc(str(r["name"])))
            .replace("{code}", r["code"])
            .replace("{mkt}", '<span class="chip kq">%s</span>' % r["mkt"] if r["mkt"] == "KOSDAQ" else r["mkt"])
            .replace("{since}", since)
            .replace("{val}", "{:,}".format(int(round(r["val"]))))
            .replace("{gap}", "%+.1f" % gap)
            .replace("{prc}", "{:,}".format(int(round(r["close"]))))
            .replace("{after}", "%+.1f" % (r["after"] * 100))
            .replace("{dir}", "up" if r["after"] > 0 else "dn")
            .replace("{nxt}", '<span class="nxt">NXT</span>' if r["nxt"] else "")
            .replace("{nxtf}", "1" if r["nxt"] else "0")
            .replace("{theme}", theme_html)
            .replace("{alerts}", alerts_html)
            .replace("{news}", news_html)
            .replace("{w}", "%.1f" % min(100.0, gap / 15 * 100)))
    gcount = {}
    for r in rows:
        gcount[r["since"]] = gcount.get(r["since"], 0) + 1
    parts = []
    prev = object()
    for r, c in zip(rows, cards):
        b = r["since"]
        if b != prev:
            lab = "오늘" if b == 0 else "%d일 전" % b
            parts.append('<div class="grp">%s<span class="gn">%d개</span></div>' % (esc(lab), gcount[b]))
            prev = b
        parts.append(c)
    rows_html = "\n".join(parts) or '<div class="empty">최근 %d거래일 내 갭 5%% · 거래대금 %d억+ 종목 없음</div>' % (WINDOW, MIN_VAL)
    mx = max((r["g"] * 100 for r in rows), default=0.0)
    warn = ('<div class="warn">⚠ 최신 갱신 실패 — 이전 결과 표시 (%s)</div>' % when) if failed else ""
    html = (PAGE
            .replace("{warn}", warn)
            .replace("{window}", str(WINDOW))
            .replace("{minval}", str(MIN_VAL))
            .replace("{total}", str(len(rows)))
            .replace("{today}", str(today))
            .replace("{nxt}", str(n_nxt))
            .replace("{maxgap}", "%+.1f%%" % mx)
            .replace("{when}", when)
            .replace("{univ}", str(nuniv))
            .replace("{cacheh}", str(CACHE_HOURS))
            .replace("{rows}", rows_html))
    with open(HTML, "w", encoding="utf-8") as f:
        f.write(html)
    if os.environ.get("GAP_NO_PUSH") != "1":
        try:
            import subprocess
            subprocess.run([sys.executable, os.path.join(HERE, "push_site.py"), HTML],
                           timeout=120, cwd=HERE)
        except Exception:
            pass
    return len(rows)


def build(force=False):
    obj, cached = get_data(force=force)
    try:
        hist = record_history(obj)
    except Exception as e:
        print("이력 기록 실패:", e)
        hist = {}
    rows, nuniv = scan(obj)
    try:
        th = get_themes(force=force)
    except Exception:
        th = {"m": {}, "size": {}}
    tmap = th.get("m", {})
    sizes = th.get("size", {})
    for r in rows:
        r["themes"] = [t for t in tmap.get(r["code"], [])
                       if not any(k in t for k in THEME_EXCL)]
    get_news(rows, force=force)
    apply_scores(rows)
    share = {}
    for r in rows:
        for t in r["themes"]:
            share[t] = share.get(t, 0) + 1
    for r in rows:
        cand = [(t, share[t]) for t in r["themes"] if share[t] >= 2]
        cand.sort(key=lambda x: (sizes.get(x[0], 999), -x[1]))
        r["shared"] = cand[:3]
    when = obj["when"] + (" (캐시)" if cached else " (신규)")
    n = render(rows, nuniv, when)
    try:
        json.dump({"n": n, "when": obj["when"]}, open(os.path.join(HERE, "gap_all_meta.json"), "w"))
    except Exception:
        pass
    return {"n": n, "when": when, "univ": nuniv, "hist": hist}


def _maybe_open():
    """HTML을 기본 프로그램으로 연다.
    os.startfile 은 Windows 전용이라 Linux(Actions)에선 아예 없으므로
    속성 확인을 하고, CI/서버에서는 건너뛴다. 실패해도 절대 예외를 밖으로 넘기지 않는다."""
    if os.environ.get("GAP_NO_OPEN") == "1" or os.environ.get("GAP_NO_PUSH") == "1":
        return False
    if not hasattr(os, "startfile"):
        return False
    try:
        os.startfile(HTML)
        return True
    except Exception:
        return False


def main():
    opened = os.path.exists(HTML) and _maybe_open()
    hist = {}
    try:
        res = build()
        n = res["n"]
        hist = res.get("hist", {})
    except Exception as e:
        print("실패:", e)
        render([], 0, _now().strftime("%Y-%m-%d %H:%M"), True)
        n = 0
    print("생성 ->", HTML, "/", n, "종목")
    if hist:
        print("이력 -> %s (%d행, 신규 %d, 이월 %d)"
              % (os.path.basename(HIST), hist.get("rows", 0),
                 hist.get("new", 0), hist.get("kept", 0)))
    if not opened:
        _maybe_open()


if __name__ == "__main__":
    main()
