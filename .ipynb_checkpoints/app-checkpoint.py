import os
import platform
import socket
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from database import init_db, insert_metric, load_metrics, count_metrics, clear_metrics
from ml_model import train_model, predict_risk

try:
    import psutil
except ImportError:
    psutil = None

st.set_page_config(page_title='NexaPC | Laptop Performance', page_icon='💻', layout='wide', initial_sidebar_state='expanded')

# -----------------------------
# Design system — based on the 4th concept: clean light SaaS dashboard
# -----------------------------
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap&#39;);
:root{--bg:#F7F9FE;--surface:#FFFFFF;--text:#172B4D;--muted:#667085;--line:#E5EAF3;--blue:#2F80ED;--blue-soft:#EAF3FF;--green:#22B573;--green-soft:#EAFBF3;--purple:#7B61FF;--purple-soft:#F1EDFF;--red:#F05D6B;--red-soft:#FFF0F2;--orange:#F5A623;--orange-soft:#FFF6E5;--nav:#F8FAFF;--nav-active:#E8F1FF}
html,body,[class*='css']{font-family:'Inter',sans-serif}
.stApp{background:var(--bg)!important;color:var(--text)!important}.block-container{max-width:1540px;padding:1.1rem 1.8rem 1.8rem}[data-testid='stHeader']{background:rgba(247,249,254,.94)}
section[data-testid='stSidebar']{background:var(--nav)!important;border-right:1px solid #E3EAF5} section[data-testid='stSidebar'] *{color:#344054!important} section[data-testid='stSidebar'] .stRadio label{border-radius:10px;padding:8px 11px;margin:2px 0} section[data-testid='stSidebar'] .stRadio label:hover{background:var(--nav-active)}
.brand{padding:5px 4px 18px}.brand-row{display:flex;align-items:center;gap:10px}.brand-icon{width:42px;height:42px;border-radius:12px;background:#EAF3FF;border:1px solid #D7E8FF;display:flex;align-items:center;justify-content:center;font-size:22px}.brand-name{font-size:19px;font-weight:800;color:#172B4D}.brand-sub{font-size:11px;color:#8492A6;margin-left:52px;margin-top:-5px}
.hero{background:linear-gradient(135deg,#FFFFFF 0%,#F8FBFF 55%,#EEF5FF 100%);border:1px solid #E0E8F5;border-radius:20px;padding:19px 24px;box-shadow:0 6px 25px rgba(38,74,128,.055);margin-bottom:16px}.hero-top{display:flex;justify-content:space-between;align-items:center;gap:15px}.hero-title{font-size:28px;font-weight:800;color:#172B4D;margin:0}.hero-sub{font-size:13px;color:#667085;margin:5px 0 0}.status{display:inline-flex;align-items:center;gap:7px;background:#EAFBF3;border:1px solid #CFF2DF;color:#168653;padding:7px 11px;border-radius:999px;font-size:12px;font-weight:700}.dot{width:7px;height:7px;background:#22B573;border-radius:50%}
.card{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 3px 16px rgba(38,74,128,.045);height:100%}.kpi-card{border-top:3px solid var(--accent,#2F80ED);background:linear-gradient(180deg,var(--soft,#FFFFFF) 0%,#FFFFFF 75%)}.card-title{font-size:13px;font-weight:700;color:#344054}.kpi{font-size:27px;font-weight:800;color:#172B4D;margin-top:5px}.kpi-meta{font-size:11px;color:#667085;margin-top:2px}.trend-up{color:#22B573;font-weight:700}.trend-down{color:#F05D6B;font-weight:700}.mini-line{height:4px;border-radius:4px;background:linear-gradient(90deg,var(--accent,#2F80ED) 0 65%,#E9EEF6 65%);margin-top:12px}.section{font-size:18px;font-weight:800;color:#172B4D;margin:18px 0 10px}.tiny{font-size:11px;color:#8492A6}.info-row{display:flex;justify-content:space-between;gap:12px;padding:8px 0;border-bottom:1px solid #EEF1F6;font-size:12px}.info-row:last-child{border-bottom:0}.info-key{color:#667085}.info-val{font-weight:700;color:#344054;text-align:right}.pill{display:inline-block;border-radius:999px;padding:4px 9px;font-size:11px;font-weight:700}.pill-green{background:#EAFBF3;color:#168653}.pill-yellow{background:#FFF6E5;color:#B7791F}.pill-red{background:#FFF0F2;color:#C63E4E}.app-row{display:grid;grid-template-columns:1.4fr .65fr .65fr .65fr;gap:10px;align-items:center;padding:9px 0;border-bottom:1px solid #EEF1F6;font-size:12px}.app-head{font-weight:700;color:#667085}.app-name{font-weight:700;color:#344054}.progress{height:6px;background:#EDF2F8;border-radius:99px;overflow:hidden}.progress>span{display:block;height:100%;background:#2F80ED;border-radius:99px}
[data-testid='stMetric']{background:#fff!important;border:1px solid var(--line)!important;border-radius:14px!important}.stButton>button{border-radius:9px!important;font-weight:700!important;border:1px solid #D5DCE8!important;background:#fff!important;color:#344054!important}.stButton>button:hover{border-color:#2F80ED!important;color:#2F80ED!important}.primary .stButton>button{background:#2F80ED!important;color:#fff!important;border-color:#2F80ED!important}[data-baseweb='select']>div,input{background:#fff!important;color:#344054!important;border-color:#D5DCE8!important}[data-testid='stDataFrame']{border:1px solid var(--line);border-radius:12px;overflow:hidden}.footer{color:#8492A6;font-size:11px;padding:14px 2px;text-align:center}
</style>
''', unsafe_allow_html=True)

init_db()

# --------------------------
# Helpers
# --------------------------
def collect_system_metrics():
    if psutil is None:
        return None
    cpu = psutil.cpu_percent(interval=0.15)
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_usage(os.path.abspath(os.sep)).percent
    bat = psutil.sensors_battery()
    battery = float(bat.percent) if bat else 100.0
    plugged = bool(bat.power_plugged) if bat else True
    temperature = 45.0
    try:
        temps = psutil.sensors_temperatures()
        values = [float(item.current) for entries in temps.values() for item in entries if item.current and item.current > 0]
        if values:
            temperature = max(values)
    except Exception:
            pass
    net = psutil.net_io_counters()
    network_mb = (net.bytes_sent + net.bytes_recv) / (1024 * 1024)
    processes = len(psutil.pids())
    uptime_hours = (time.time() - psutil.boot_time()) / 3600
    return
{'cpu':cpu,'ram':ram,'disk':disk,'temperature':temperature,'battery':battery,'network_mb':network_mb, 'processes':processes, 'uptime_hours':uptime_hours, 'plugged':plugged,}

def demo_metric(rng, ts=None):
    t = time.time() if ts is None else ts
    wave = np.si(t / 2600)
    return {'cpu':float(np.clip(38 + 18*wave + rng.normal(0,7),5,95)), 'ram':float(np.clip(44+11*wave + rng.normal(0,4),10,92)), 'disk':float(np.clip(50 + rng.normal(0,1.2),10,95)), 'temperature':float(np.clip(54+7*wave+rng.normal(0,3),35,92)), 'battery':float(np.clip(82 -(time.time()-t)/3600 if ts is None else 82 + rng.normal(0,2),10,100)), 'netwrok_mb':float(rng.uniform(100,900)), 'processes':int(rng.integers(90,210)), 'uptime_hours':float(rng.uniform(1,72)), 'plugged':bool(rng.choice([True, False], p=[.65,.35]))}


def performance_score(row):
    score=100
    score-=max(0,row['cpu']-50)*.25
    score-=max(0,row['ram']-60)*.35
    score-=max(0,row['disk']-80)*.18
    score-=max(0,row['temperature']-65)*.45
    score-=max(0,25-row['battery'])*.15
    return float(np.clip(score,0,100))


def score_status(score):
    if score>=80:return 'Excellent', 'pill-green'
    if score>=65:return 'Good', 'pill-green'
    if score>=50:return 'Needs Attention', 'pill-yellow'
    return 'Critical', 'pill-red'


def pct_delta(current, baseline):
    if baseline in (None, 0): return 0
    return ((current-baseline)/abs(baseline))*100


def chart_theme(fig, height=300):
    fig.update_layout(height=height,paper_bgcolor='#FFFFFF',plot_bgcolor='#FFFFFF',font= {'family':'Inter, sans-serif','color':'#344054','size':11},margin= {'l':45,'r':20,'t':45,'b':40},legend={'bgcolor':'#FFFFFF','font':{'color':'#475467'}},hoverlabel= {'bgcolor':'#FFFFFF','font':{'color':'#101828'}},colorway= ['#2F80ED','#22B573','#7B61FF','#F05D6B','#F5A63'],xaxis= {'gridcolor':'#EEF2F7','linecolor':'#DCE3ED','tickfont':{'color':'#667085'}},yaxis= {'gridcolor':'#EEF2F7','linecolor':'#DCE3ED','tickfont':{'color':'#667085'}})
    return fig


def metric_card(title,value,meta,delta=None,accentt='blue'):
    accent={'blue':('#2F80ED','#EAF3FF'),'green':('#22B573','#EAFBF3','#EAFBF3'),'purple': ('#7B61FF','#F1EDFF'),'red':('#F05D6B','#FFF0F2'),'orange':('#F5A623','#FFF6E5')}
    a,soft=accents.get(accent,accents['blue'])
    d=''
    if delta is not None:
        cls='trend-up' if delta<=0 else 'trend-down'; arrow='↓' if delta<=0 else '↑'
        d=f'<span class="{cls}">{arrow} {abs(delta):.1f}%</span> vs recent avg'
    st.markdown(f'<div class="card kpi-card" style="--accent:{a};--soft:{soft}"><div class="kpi-title">{title}</div><div class="kpi">{value}</div><div class="kpi-meta">{meta}</ddiv><div class="kpi-meta">{d}</div><div class="mini-line"></div></div>',unsafe_allow_html=True)


def gauge(value,title,number_suffix='%'):
    fig=go.Figure(go.indicator(mode='gauge+number',value=value,number={'suffix':number_suffix,'font': {'size':27,'color':'#10213F'}},title={'text':title,'font':{'size':12,'color':'#667085'}},gauge= {'axis':{'range':[0,100],'tickfont':{'color':'#667085','size':9}},'bar': {'color':'#2F80ED','thickness':.7},,'bgcolor':'#EEF2F7','borderwidth':0,'steps':[{'range':[0,60], 'color':'#f2f4f7'},{'range':[60,80],'color':'#EAF3FF'},{'range':[80,100],'color':'#EAFBF3'}]}))
    fig.update_layout(height=185,paper_bgcolor,='white',margin={'l':10,'r':10,'t':35,'b':5})
    return fig


def add_demo_history():
    rng=np.random.default_rng(7)
    now=datetime.now()
    for i in range(96,0,-1):
        ts=now-timedelta(minutes=15*i)
        insert_metric(demo_metric(rng,ts.timestamp()),timestamp=ts.strftime('%y-%m-%d %H:%M:%S'))


def load_df():
    df=load_metrics(1500)
    if df.empty:
        return df
    df['timestmap']=pd.to_datetime(df['timestamp'])
    df['score']=df.apply(performance_score,axis=1)
    return df


def top_processes():
    if psutil is None:return pd.DataFrame()
        rows=[]
    for p in psutil.process_iter(['name','cpu_percent','memory_percent']):
        try:
            info=p.info
            rows.append({'Application':info.get('name') or 'Unknown','CPU %':float(info.get('cpu_percent') or 0),'Memory %':float(info.get('memory_percent') or 0)})
        except Exception: pass
    out=pd.DataFrame(rows)
    if out.empty:return out
    return out.groupby('Application',as_index=False).agg({'CPU %':'max','Memory %':'max'}).sort_values('CPU %',ascending=False).head(8)

# Capture one sample per rerun
live=collect_system_metrics()
if live is None:
    rng=np.random.default_rng()
    live=demo_metric(rng)
insert_metric(live)
df=load_df()
score=performance_score(live)
status,status_cls=score_status(score)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-row"><div class="brand-icon">💻</div><div class="brand-name">NexaPC</div></div><div class="brand-sub">Powering productivity every day</div></div>',unsafe_allow_html=True)
    page=st.radio('Workspace',['Dashboard','Performance','Applications','Battery & Power','Storage & Health','ML & Alerts'],label_visibility='collapsed')
    st.divider()
    st.markdown('**Monitor**')
    auto=st.toggle('Live refresh',value=True)
    interval=st.slider('Refresh interval',5,60,15,step=5,format='%d sec')
    st.divider()
    st.markdown('**Device**')
    st.caption(platform.system()+' '+platform.release())
    st.caption(socket.gethostname())
    st.caption(f'{count_metrics():,} telemetry records')
    if st.button('↻ Refresh now',use_container_width=True):
        st.rerun()
    if st.button('Clear telemetry',use_container_width=True):
        clear_metrics(); st.rerun()

# -----------------------------
# Header
# -----------------------------
header_left,header_mid,header_right=st.columns([5.2,2.3,1.2])
with header_left:
    st.markdown('<div style="font-size:25px;font-weight:800;color:#10213f">Laptop Performance Dashboard</div><div style="font-size:12px;color:#667085">Monitor. Manage. Maximize.</div>',unsafe_allow_html=True)
with header_mid:
    date_label=st.selectbox('Period',['Last 24 Hours','Last 7 Days','Last 30 Days'],label_visibility='collapsed')
with header_right:
    st.markdown(f'<div style="text-align:right;padding-top:7px"><span class="status"><span class="dot"></span>{status}</span></div>',unsafe_allow_html=True)

# -----------------------------
# Dashboard
# -----------------------------
if page=='Dashboard':
    st.markdown('<div class="section">System Overview</div>',unsafe_allow_html=True)
    recent=df.tail(20) if not df.empty else pd.DataFrame()
    avg=recent.mean(numeric_only=True) if not recent.empty else pd.Series()
    cols=st.columns(5)
    with cols[0]: metric_card('Performance Score',f'{score:.0f}/100',status)
    with cols[1]: metric_card('CPU',f'{live["cpu"]:.0f}%','Current utilization',pct_delta(live['cpu'],avg.get('cpu',live['cpu'])))
    with cols[2]: metric_card('Memory',f'{live["ram"]:.0f}%','RAM utilization',pct_delta(live['ram'],avg.get('ram',live['ram'])))
    with cols[3]: metric_card('Storage',f'{live["disk"]:.0f}%','Disk capacity used',pct_delta(live['disk'],avg.get('disk',live['disk'])))
    with cols[4]: metric_card('Battery',f'{live["battery"]:.0f}%','Estimated remaining level',pct_delta(live['battery'],avg.get('battery',live['battery'])))

    st.markdown('<div class="section">Resource Utilization</div>',unsafe_allow_html=True)
    c1,c2=st.columns([1.75,1])
    with c1:
        plot=df.tail(80).copy()
        fig=px.line(plot,x='timestamp',y=['cpu','ram','disk'],labels={'value':'Utilization %','timestamp':'','variable':''})
        fig=chart_theme(fig,325); fig.update_layout(title={'text':'CPU, Memory & Disk','x':.02,'font':{'size':16,'color':'#10213f'}})
        st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    with c2:
        st.markdown('<div class="card"><div class="card-title">System Health</div>',unsafe_allow_html=True)
        st.plotly_chart(gauge(score,'Overall Health'),use_container_width=True,config={'displayModeBar':False})
        checks=[('CPU',live['cpu']<85),('Memory',live['ram']<85),('Storage',live['disk']<90),('Battery',live['battery']>20 or live['plugged']),('Temperature',live['temperature']<80)]
        for label,ok in checks:
            st.markdown(f'<div class="info-row"><span class="info-key">{label}</span><span class="pill {"pill-green" if ok else "pill-red"}">{"Normal" if ok else "Attention"}</span></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="section">Application Usage & Device Information</div>',unsafe_allow_html=True)
    a,b=st.columns([1.35,1])
    with a:
        proc=top_processes()
        st.markdown('<div class="card"><div class="card-title">Top Resource-Using Applications</div>',unsafe_allow_html=True)
        if proc.empty:
            st.info('Application-level telemetry is not available on this system.')
        else:
            st.dataframe(proc.rename(columns={'CPU %':'CPU','Memory %':'Memory'}).style.format({'CPU':'{:.1f}%','Memory':'{:.1f}%'}),use_container_width=True,height=300,hide_index=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><div class="card-title">System Information</div>',unsafe_allow_html=True)
        info=[('Device',socket.gethostname()),('Processor',platform.processor() or 'System CPU'),('OS',platform.platform()),('RAM',f'{psutil.virtual_memory().total/(1024**3):.1f} GB' if psutil else 'N/A'),('Uptime',f'{live["uptime_hours"]:.1f} hours'),('Power','AC connected' if live['plugged'] else 'Battery')]
        for k,v in info: st.markdown(f'<div class="info-row"><span class="info-key">{k}</span><span class="info-val">{v}</span></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

# -----------------------------
# Performance
# -----------------------------
elif page=='Performance':
    st.markdown('<div class="section">Performance Analytics</div>',unsafe_allow_html=True)
    days={'Last 24 Hours':1,'Last 7 Days':7,'Last 30 Days':30}[date_label]
    hist=df[df['timestamp']>=datetime.now()-timedelta(days=days)].copy()
    c1,c2=st.columns(2)
    with c1:
        fig=px.line(hist,x='timestamp',y='score'); fig=chart_theme(fig,310); fig.update_layout(title={'text':'Performance Score Trend','font':{'size':16,'color':'#10213f'}}); st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    with c2:
        fig=px.line(hist,x='timestamp',y='temperature'); fig=chart_theme(fig,310); fig.update_layout(title={'text':'Temperature Trend','font':{'size':16,'color':'#10213f'}}); st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    fig=px.area(hist,x='timestamp',y=['cpu','ram','disk']); fig=chart_theme(fig,330); fig.update_layout(title={'text':'Resource Utilization Over Time','font':{'size':16,'color':'#10213f'}}); st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    st.markdown('<div class="section">Performance Statistics</div>',unsafe_allow_html=True)
    summary=hist[['cpu','ram','disk','temperature','battery','score']].describe().T.round(2).reset_index().rename(columns={'index':'Metric'})
    st.dataframe(summary,use_container_width=True,hide_index=True)

# -----------------------------
# Applications
# -----------------------------
elif page=='Applications':
    st.markdown('<div class="section">Application Performance</div>',unsafe_allow_html=True)
    proc=top_processes()
    c1,c2,c3=st.columns(3)
    with c1: metric_card('Running Processes',f'{live["processes"]:,}','Current process count',accent='blue')
    with c2: metric_card('CPU Load',f'{live["cpu"]:.1f}%','System-wide utilization',accent='blue')
    with c3: metric_card('Memory Load',f'{live["ram"]:.1f}%','System-wide memory',accent='green')
    if not proc.empty:
        fig=px.bar(proc.sort_values('CPU %'),x='CPU %',y='Application',orientation='h',text='CPU %'); fig=chart_theme(fig,360); fig.update_layout(title={'text':'Top Applications by CPU Usage','font':{'size':16,'color':'#10213f'}},xaxis_title='CPU %',yaxis_title=''); st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
        st.dataframe(proc.style.format({'CPU %':'{:.1f}%','Memory %':'{:.1f}%'}),use_container_width=True,hide_index=True)
    else: st.warning('psutil is unavailable; install the project requirements to enable application monitoring.')

# -----------------------------
# Battery & Power
# -----------------------------
elif page=='Battery & Power':
    st.markdown('<div class="section">Battery & Power Center</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown('<div class="card">',unsafe_allow_html=True); st.plotly_chart(gauge(live['battery'],'Battery Level'),use_container_width=True,config={'displayModeBar':False}); st.markdown(f'<div style="text-align:center;font-weight:700;color:#344054">{"Charging / AC Connected" if live["plugged"] else "Running on Battery"}</div></div>',unsafe_allow_html=True)
    with c2: metric_card('Estimated Runtime',f'{max(0,live["battery"]/22):.1f} h','Approximation based on current level',accent='green')
    with c3: metric_card('Power Mode','Balanced','Recommended everyday profile',accent='orange')
    hist=df.tail(120)
    fig=px.line(hist,x='timestamp',y='battery'); fig=chart_theme(fig,330); fig.update_layout(title={'text':'Battery Level Trend','font':{'size':16,'color':'#10213f'}}); st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    st.markdown('<div class="card"><div class="card-title">Power Recommendations</div>',unsafe_allow_html=True)
    rec=['Keep the charger connected for sustained high-performance workloads.' if live['plugged'] else 'Connect the charger before battery falls below 20%.','Use Balanced mode for everyday productivity.','Reduce screen brightness and background applications when on battery.']
    for r in rec: st.markdown(f'• {r}')
    st.markdown('</div>',unsafe_allow_html=True)

# -----------------------------
# Storage & Health
# -----------------------------
elif page=='Storage & Health':
    st.markdown('<div class="section">Storage & Device Health</div>',unsafe_allow_html=True)
    free=100-live['disk']
    c1,c2=st.columns([1,1.4])
    with c1:
        fig=go.Figure(go.Pie(labels=['Used','Free'],values=[live['disk'],free],hole=.68,textinfo='none',marker={'colors':['#7B61FF','#E7EAF2']})); fig.update_layout(height=300,paper_bgcolor='white',showlegend=True,legend={'font':{'color':'#475467'}}); fig.add_annotation(text=f'<b>{live["disk"]:.0f}%</b><br>Used',x=.5,y=.5,showarrow=False,font={'size':18,'color':'#10213f'}); st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    with c2:
        st.markdown('<div class="card"><div class="card-title">Health Checklist</div>',unsafe_allow_html=True)
        checks=[('Storage capacity',live['disk']<80,'Healthy' if live['disk']<80 else 'Free space recommended'),('Thermal state',live['temperature']<70,'Normal' if live['temperature']<70 else 'Warm'),('Battery condition',live['battery']>30 or live['plugged'],'Healthy' if live['battery']>30 or live['plugged'] else 'Low'),('System uptime',live['uptime_hours']<168,'Normal' if live['uptime_hours']<168 else 'Restart recommended')]
        for k,ok,msg in checks: st.markdown(f'<div class="info-row"><span class="info-key">{k}</span><span class="pill {"pill-green" if ok else "pill-yellow"}">{msg}</span></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    fig=px.line(df.tail(150),x='timestamp',y=['disk','temperature']); fig=chart_theme(fig,320); fig.update_layout(title={'text':'Storage & Thermal Trend','font':{'size':16,'color':'#10213f'}}); st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})

# -----------------------------
# ML & Alerts
# -----------------------------
else:
    st.markdown('<div class="section">ML Intelligence & Alerts</div>',unsafe_allow_html=True)
    model,accuracy=train_model(df)
    pred=predict_risk(model,live)
    c1,c2,c3,c4=st.columns(4)
    with c1: metric_card('Slowdown Risk',f'{pred["slowdown_risk"]:.0f}%','ML predicted probability',accent='orange')
    with c2: metric_card('Overheat Risk',f'{pred["overheat_risk"]:.0f}%','Thermal risk estimate',accent='red')
    with c3: metric_card('Health Score',f'{score:.0f}/100',status,accent='blue')
    with c4: metric_card('Model Accuracy',f'{accuracy:.0f}%','Validation estimate',accent='purple')
    a,b=st.columns([1,1])
    with a:
        st.markdown('<div class="card"><div class="card-title">Prediction Inputs</div>',unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{'CPU %':live['cpu'],'RAM %':live['ram'],'Disk %':live['disk'],'Temp °C':live['temperature'],'Battery %':live['battery'],'Processes':live['processes']}]).round(2),use_container_width=True,hide_index=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><div class="card-title">Active Alerts</div>',unsafe_allow_html=True)
        alerts=[]
        if live['cpu']>=85: alerts.append(('pill-red','High CPU','CPU utilization is above 85%.'))
        if live['ram']>=85: alerts.append(('pill-red','High memory','Memory pressure may affect applications.'))
        if live['temperature']>=80: alerts.append(('pill-red','High temperature','Cooling should be checked.'))
        if live['disk']>=90: alerts.append(('pill-yellow','Low storage','Free disk space to maintain performance.'))
        if live['battery']<=20 and not live['plugged']: alerts.append(('pill-yellow','Low battery','Connect the charger soon.'))
        if not alerts: st.markdown('<div class="pill pill-green">All systems normal</div><p style="color:#667085;font-size:12px">No critical performance alerts detected.</p>',unsafe_allow_html=True)
        else:
            for cls,title,msg in alerts: st.markdown(f'<div style="padding:9px 0;border-bottom:1px solid #f0f2f5"><span class="pill {cls}">{title}</span><div class="tiny" style="margin-top:4px">{msg}</div></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('<div class="section">Recommended Actions</div>',unsafe_allow_html=True)
    recommendations=[]
    if live['cpu']>70: recommendations.append('Close unnecessary CPU-intensive applications.')
    if live['ram']>70: recommendations.append('Close unused browser tabs and memory-heavy applications.')
    if live['temperature']>70: recommendations.append('Improve airflow and check the cooling system.')
    if live['battery']<30 and not live['plugged']: recommendations.append('Connect the laptop to AC power.')
    if live['disk']>80: recommendations.append('Clean temporary files and free storage.')
    if not recommendations: recommendations=['Continue normal usage; no optimization is currently required.']
    for r in recommendations: st.markdown(f'✅ {r}')

# Footer and downloads
st.divider()
d1,d2,d3=st.columns([1,1,4])
with d1:
    csv=df.to_csv(index=False).encode('utf-8')
    st.download_button('Export CSV',csv,'laptop_telemetry.csv','text/csv',use_container_width=True)
with d2:
    if st.button('Generate demo history',use_container_width=True):
        add_demo_history(); st.rerun()
with d3:
    st.markdown(f'<div class="footer">Last captured: {datetime.now().strftime("%d %b %Y • %I:%M:%S %p")} &nbsp; • &nbsp; NexaPC Performance Analytics &nbsp; • &nbsp; {count_metrics():,} records</div>',unsafe_allow_html=True)

if auto:
    time.sleep(interval)
    st.rerun()

    