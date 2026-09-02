# Food Delivery Time Prediction

Flask + scikit-learn web app for predicting food delivery time.

## Deployment on Render

1. Create a GitHub repository named `Food_Delivery_Time_Prediction`.
2. Upload the contents of this folder to the repository root.
3. On Render, create a **Web Service** from that GitHub repository.
4. Use:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Plan: Free
   - Root Directory: leave blank
5. Deploy.

`render.yaml` contains the same settings for a Blueprint deployment.

## Local run

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.

## Model

The deployment model is a resource-friendly Random Forest with 50 trees and `max_depth=20`. It is intentionally much smaller than the previous unlimited-depth model so the Flask process can fit within a small free instance's memory budget.
