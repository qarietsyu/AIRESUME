# AI Resume ATS Checker

Upload a resume (PDF, DOCX or TXT) and get an ATS score, a score breakdown, missing keywords,
prioritized improvements and example rewrites. Optionally paste a job description for better keyword matching.

Built with **Streamlit** and **Google Gemini Flash**.

## Features
- Overall ATS score (0-100) with breakdown: keywords, formatting, content quality, readability
- Strengths, weaknesses and missing keywords
- Prioritized, actionable improvement suggestions
- Before/after rewrite examples for weak resume lines
- Download the report as JSON

## Run locally
```bash
git clone https://github.com/<your-username>/ai-resume-assistnt.git
cd ai-resume-assistnt
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"      # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```
Get a free API key at https://aistudio.google.com/apikey. You can also paste the key into the app sidebar.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (`app.py`, `requirements.txt`, `README.md` in the root).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose the repo, branch `main`, main file `app.py`.
4. Open **Advanced settings > Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**.

## Configuration
The model defaults to `gemini-2.5-flash`. Change it in the app sidebar if Google releases a newer Flash model.

## Notes
- Scanned or image-only PDFs have no extractable text. The app will warn you, and real ATS software cannot read them either.
- Never commit your API key to GitHub.
