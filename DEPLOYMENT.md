# SupplyPulse AI - Production Deployment Guide

This guide outlines the step-by-step architecture and procedure to take the SupplyPulse AI intelligence platform from a local development environment to a live, production-grade cloud infrastructure.

## 🏗️ Target Production Architecture

To support the heavy Machine Learning requirements (XGBoost, SHAP) while keeping the frontend lightning fast, we recommend a decoupled, multi-cloud approach:

1. **Frontend (Next.js)** -> **Vercel**: Vercel provides world-class global Edge caching, seamless CI/CD, and zero-config Next.js optimizations.
2. **Backend (FastAPI & ML Engine)** -> **Render (or Railway)**: Supabase Edge Functions run TypeScript/Deno, which cannot natively run large Python XGBoost/Pandas ML models. Therefore, the FastAPI backend must be hosted on a dedicated Python PaaS like Render or Railway.
3. **Database (PostgreSQL)** -> **Supabase**: We will migrate the in-memory `test.csv` dataset into a robust Supabase PostgreSQL database to handle enterprise-scale logistics queries.

---

## Phase 1: Version Control (GitHub)

Before deploying to any cloud provider, the code must be pushed to a Git repository.

1. **Initialize Git**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: SupplyPulse AI Production Release"
   ```
2. **Create GitHub Repo**: Go to GitHub, create a new private/public repository named `supplypulse-ai`.
3. **Push Code**:
   ```bash
   git remote add origin https://github.com/yourusername/supplypulse-ai.git
   git branch -M main
   git push -u origin main
   ```

---

## Phase 2: Database Setup (Supabase)

Currently, the backend reads from `test.csv`. For production, this data needs to be in a real database.

1. **Create Supabase Project**: Go to [Supabase](https://supabase.com/) and create a new project.
2. **Migrate Dataset**: 
   * Go to the Supabase **Table Editor**.
   * Click **Create a new table** (name it `shipments`).
   * Import your `test.csv` directly into the table via the Supabase UI.
3. **Get Credentials**: Go to Project Settings -> Database. Copy your `Database Connection URI`.
4. **Update Backend Code (Optional but Recommended)**: 
   * Install `psycopg2-binary` and `sqlalchemy` in the backend.
   * Modify `backend/repositories/shipment_repository.py` to query Supabase instead of reading the CSV.

---

## Phase 3: Backend Deployment (Render)

Render is ideal for Python ML backends.

1. **Prepare Backend**:
   * Ensure `backend/requirements.txt` includes everything (`fastapi`, `uvicorn`, `pandas`, `xgboost`, `shap`, `scikit-learn`).
   * Add a `render.yaml` or just configure it via the Render Dashboard.
2. **Deploy**:
   * Go to [Render](https://render.com/).
   * Click **New+** -> **Web Service**.
   * Connect your GitHub repository.
   * **Root Directory**: `backend` (if you separate them) or leave blank and set the build command.
   * **Build Command**: `pip install -r backend/requirements.txt`
   * **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   * **Environment Variables**: Add your `DATABASE_URL` (from Supabase).
3. **Verify**: Once deployed, Render will give you a URL (e.g., `https://supplypulse-api.onrender.com`). Verify that `https://supplypulse-api.onrender.com/docs` works.

---

## Phase 4: Frontend Deployment (Vercel)

Vercel will host the Next.js UI and automatically connect to your live Python API.

1. **Prepare Frontend**:
   * Open `frontend/src/lib/api.ts`.
   * Ensure the `API_URL` reads from an environment variable:
     ```typescript
     const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000/api/v1";
     ```
2. **Deploy to Vercel**:
   * Go to [Vercel](https://vercel.com/).
   * Click **Add New Project** and import the `supplypulse-ai` GitHub repository.
   * **Framework Preset**: Next.js.
   * **Root Directory**: Select the `frontend` folder! This is critical since it's a monorepo.
3. **Set Environment Variables**:
   * Add `NEXT_PUBLIC_API_URL` = `https://supplypulse-api.onrender.com/api/v1` (the Render URL from Phase 3).
4. **Deploy**: Click Deploy. Vercel will build and assign you a live `.vercel.app` domain.

---

## Final Checklist & Next Steps

- [ ] **CORS Configuration**: Ensure your FastAPI backend has CORS configured to accept requests from your new `https://your-project.vercel.app` domain.
- [ ] **ML Artifacts**: Ensure your pre-trained `.pkl` or `.json` XGBoost models are committed to GitHub so Render can load them into memory on startup.
- [ ] **Security**: Secure the Supabase database with Row Level Security (RLS) if you plan on querying it directly from the frontend in the future.
