# Turkey Creek: CRB3 calibrations on pairs of CPMs

Streamlit app comparing 10 NSGA-II calibrations of the UMN HLM CRB3 runoff model for Turkey Creek
(USGS 02250030), each on two composite performance metrics (CPMs), plus the default parameters, on 2017.

- `app.py`: the app (reads only `data/`). Run locally: `pip install -r requirements.txt && streamlit run app.py`
- `data/`: exported results (hourly Q / mean(Q), basin-average soil and groundwater storage, metrics,
  NSGA-II evaluations, parameter ranges, CPM weights).
- `export_data.py`: makes `data/` from the model outputs (run on the cluster; not used by the app).

Deployed on Streamlit Community Cloud from this repository (main file `app.py`, Python 3.12).
