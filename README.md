# NexaPC — Laptop Performance Dashboard

A polished Streamlit laptop-performance analytics application based on the **4th UI concept** from the supplied design options: clean light SaaS layout, compact navigation, KPI cards, resource charts, application usage, battery/storage health, system information, and ML alerts.

## Included
- Live CPU, RAM, disk, temperature, battery, process count, network and uptime telemetry
- SQLite persistence
- Interactive Plotly analytics
- Application/process resource table
- Battery & power center
- Storage & device health
- ML slowdown and overheating risk
- Alerts and recommendations
- CSV export
- Demo-history generator
- Automatic refresh with configurable interval
- Contrast-safe light UI

## Project structure
```text
laptop_performance_dashboard/
├── app.py
├── database.py
├── ml_model.py
├── laptop_performance.db
├── requirements.txt
├── README.md
├── run.bat
└── .gitignore
```

## Run on Windows
1. Open Command Prompt/PowerShell in this folder.
2. Create a virtual environment (recommended): `python -m venv .venv`
3. Activate it: `.venv\\Scripts\\activate`
4. Install packages: `pip install -r requirements.txt`
5. Start: `streamlit run app.py`

Or double-click `run.bat` after installing Python and the requirements.

## Notes
Temperature sensors are hardware/OS dependent. If unavailable, the dashboard uses a safe fallback value. Application-level telemetry depends on `psutil` permissions and OS support.

The ML labels are engineered from telemetry thresholds/weights; the model is intended as a dashboard demonstration rather than a medical, safety, or hardware-diagnostic system.
