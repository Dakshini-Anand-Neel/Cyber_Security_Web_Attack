# Deploying CyberShield — step by step

This app needs **your** GitHub account and **your** Streamlit Community Cloud
account to go live (I can't create either on your behalf, but every command
below is copy-pasteable).

## 1. Push this project to GitHub

```bash
cd cybershield          # the folder containing this file
git init
git add .
git commit -m "CyberShield: web attack detection app"
git branch -M main

# Create an empty repo first at https://github.com/new (do NOT initialize
# it with a README/license — this folder already has one), then:
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

If you'd rather I do the `git` steps for you inside this chat, share:
- the GitHub repo URL (must already exist, empty), and
- a fine-grained Personal Access Token with `contents: write` on that repo
  (create one at https://github.com/settings/tokens) — paste it only in this
  chat, and revoke it afterwards from that same settings page once we're done.

## 2. Deploy on Streamlit Community Cloud (free, ~2 minutes)

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. Click **"New app"**.
3. Repository: `<your-username>/<your-repo>`, Branch: `main`.
4. **Main file path:** `app/cs_app/app.py`
5. Click **Deploy**.

Streamlit installs `app/cs_app/requirements.txt` automatically and gives you
a public URL in the form:

```
https://<your-app-name>-<random-id>.streamlit.app
```

That URL is what you send people to use the app. Every push to `main`
auto-redeploys it.

## 3. No dataset/model needed to demo it

The deployed app works immediately with **no files or secrets** — open the
**🧪 Hypothetical Data** page and click **Generate & Train**. If you later
want it to use your *real* trained model instead, commit
`models/final_model.joblib`, `models/scaler.joblib`,
`models/target_encoder.joblib`, and `models/metadata.json` to the repo (these
are currently git-ignored — remove those lines from `.gitignore` if you want
them tracked, since Streamlit Cloud has no separate way to upload files).

## Alternative: Hugging Face Spaces (also free)

1. Create a Space at https://huggingface.co/new-space → SDK: **Streamlit**.
2. Push the same repo to the Space's git remote (HF gives you the URL after
   creation), or upload files via the web UI.
3. Set the Space's app file to `app/cs_app/app.py` in Space settings.
4. URL will be `https://huggingface.co/spaces/<you>/<space-name>`.
