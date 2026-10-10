# Public Demo Deployment

Target: Streamlit Community Cloud, Python 3.11.

Live app: https://water-watch-ukddepshzllqushgvdpzvv.streamlit.app/
Anonymous desktop/mobile startup, charts and navigation checked 10 October 2026.

## Deployment Settings

- Repository: `shivam3746/water-watch`
- Branch: `main`
- Main file path: `deployment/streamlit_app.py`
- Advanced settings / Python version: `3.11`
- No API keys or secrets are required.

Sign into https://share.streamlit.io and connect the GitHub account that owns
the repository. Select Create app, choose the repository/branch/main file above,
set Python 3.11, and deploy. OAuth account permissions must be reviewed and
accepted by the account owner. Do not enter tokens into the chat or repository.

The requirements next to the deployment entrypoint pin the tested runtime
packages; the full research dependencies remain at the repository root.
PyArrow is pinned to 24.0.0, matching the working Cloud runtime rather than
relying on the host's automatic replacement of 25.0.1.
The entrypoint verifies the bundle hashes before rendering. It does not
download datasets, load model pickles, refit models, or evaluate labels.

## Public Data and Privacy

`demo_bundle/` is an explicitly selected, hash-manifested bundle of development
results, held-out-year evaluation, plots, and short explanation traces. Dataset
license and attribution are in `DATA_ATTRIBUTION.md`; bundled dataset-derived
content is CC BY 4.0. Code remains Apache 2.0. Generated research artifacts and
raw inputs remain ignored by Git.

Public review workspaces are temporary and unique to each Streamlit session.
Users cannot see another session's decisions through the app. Reconnecting,
refreshing, sleeping/restarting the app or resource cleanup may discard reviews.
Use an alias and synthetic rationale; this is not authenticated operator logging.
The local `app.py` workflow still uses persistent SQLite storage. Hosted reviews
execute no physical action and are not copied into the bundle.

## Local Deployment Smoke Check

```powershell
.\.venv\Scripts\python.exe -m streamlit run deployment/streamlit_app.py --server.address 127.0.0.1 --server.port 8502
.\.venv\Scripts\python.exe scripts/check_dashboard.py --url http://127.0.0.1:8502
```

A fresh checkout needs only `deployment/requirements.txt` to run this entrypoint;
no local `data/` or `artifacts/` experiment outputs are necessary. In contrast,
running root `app.py` is the local research workflow and expects those outputs.

To recreate the curated bundle after an explicitly reviewed new experiment,
run `scripts/package_demo.py`. It refuses to silently overwrite an existing
bundle. Preserve the old bundle and provenance before replacing it; do not
retune against 2019 and keep calling those results an untouched final test.

## Release Verification

- Cloud build succeeds and all seven tabs load.
- Headline warnings retain the held-out result: 2/19 new events, 10.5% recall.
- Detector/month changes and event windows work.
- Plots remain readable on desktop/mobile.
- Reviews require an explicit named-alias decision; independent browser sessions
  cannot see or change one another's review history.
- Downloads contain only expected results or the current session's decisions.
- A signed-out visitor can access the public URL.
- Do not announce a public URL until it has actually deployed and been checked.

Reference: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
