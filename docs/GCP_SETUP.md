# GCP & GitHub Actions Setup

Step-by-step guide to reproduce the Cloud Run GPU deployment from scratch.

## Prerequisites

- [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed and authenticated (`gcloud auth login`)
- A GCP project created via the console
- A GitHub repository with the code

Replace the following placeholders throughout this guide:

| Placeholder | Description |
|---|---|
| `YOUR_PROJECT_ID` | GCP project ID |
| `YOUR_REGION` | GCP region (e.g. `us-east4`) |
| `YOUR_GITHUB_ORG` | GitHub username or org |
| `YOUR_GITHUB_REPO` | GitHub repository name |

---

## 1. Enable GCP APIs

```bash
gcloud config set project YOUR_PROJECT_ID

gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  cloudresourcemanager.googleapis.com \
  sts.googleapis.com
```

---

## 2. Artifact Registry

Create a Docker repository to store the container image:

```bash
gcloud artifacts repositories create spam-classifier-repo \
  --repository-format=docker \
  --location=YOUR_REGION \
  --description="Spam classifier Docker images"
```

---

## 3. GCS Bucket for Model Weights

The trained model weights are gitignored and must be stored in GCS so GitHub Actions can pull them during `docker build`.

```bash
# Create the bucket
gsutil mb -p YOUR_PROJECT_ID -l YOUR_REGION gs://YOUR_PROJECT_ID-model

# Upload your locally trained weights
gsutil -m cp -r src/model/bge_spam_classifier gs://YOUR_PROJECT_ID-model/bge_spam_classifier

# Verify
gsutil ls gs://YOUR_PROJECT_ID-model/bge_spam_classifier/
```

You should see: `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`.

---

## 4. Service Account

```bash
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions deployer"
```

Grant the three required roles:

```bash
# Push images to Artifact Registry
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

# Deploy to Cloud Run
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.developer"

# Impersonate the compute service account during deploy
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
```

Grant GCS read access for downloading model weights:

```bash
gsutil iam ch \
  serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com:objectViewer \
  gs://YOUR_PROJECT_ID-model
```

---

## 5. Workload Identity Federation (keyless auth)

No JSON keys — GitHub Actions authenticates via OIDC.

```bash
# Create the pool
gcloud iam workload-identity-pools create github-pool \
  --location=global \
  --display-name="GitHub Actions pool"

# Create the provider
gcloud iam workload-identity-pools providers create-oidc github-provider \
  --workload-identity-pool=github-pool \
  --location=global \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='YOUR_GITHUB_ORG/YOUR_GITHUB_REPO'"

# Allow the provider to impersonate the service account
gcloud iam service-accounts add-iam-policy-binding \
  github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/$(gcloud projects describe YOUR_PROJECT_ID --format='value(projectNumber)')/locations/global/workloadIdentityPools/github-pool/attribute.repository/YOUR_GITHUB_ORG/YOUR_GITHUB_REPO"
```

Get the provider resource name for the GitHub secret:

```bash
gcloud iam workload-identity-pools providers describe github-provider \
  --workload-identity-pool=github-pool \
  --location=global \
  --format="value(name)"
```

---

## 6. GitHub Secrets

Add these in your GitHub repo under **Settings → Secrets and variables → Actions**:

| Secret | Value |
|---|---|
| `WIF_PROVIDER` | Output of the `providers describe` command above |
| `WIF_SERVICE_ACCOUNT` | `github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com` |
| `API_KEY` | A strong random secret key of your choice |

---

## 7. Update the Workflow

In `.github/workflows/deployment.yml`, set the `env` block to match your project:

```yaml
env:
  PROJECT_ID: YOUR_PROJECT_ID
  REGION: YOUR_REGION
  REPOSITORY: spam-classifier-repo
  IMAGE: spam-classifier
```

Also update the GCS path in the "Download model weights" step:

```yaml
- name: Download model weights from GCS
  run: |
    mkdir -p src/model/bge_spam_classifier
    gsutil -m cp -r gs://YOUR_PROJECT_ID-model/bge_spam_classifier/* src/model/bge_spam_classifier/
```

---

## 8. First Deploy

Push to `main` to trigger the pipeline:

```bash
git push origin main
```

The workflow will:
1. Authenticate to GCP via WIF
2. Download model weights from GCS
3. Build and push the Docker image to Artifact Registry
4. Deploy to Cloud Run (NVIDIA L4, 4 CPU, 16Gi memory)
5. Print the Cloud Run URL
6. Prune old images from Artifact Registry

---

## 9. Make the Service Public

After the first successful deploy, grant public invoke access:

```bash
gcloud run services add-iam-policy-binding spam-classifier \
  --region=YOUR_REGION \
  --member=allUsers \
  --role=roles/run.invoker
```

---

## 10. Validate

```bash
# Liveness check
curl https://YOUR_CLOUD_RUN_URL/ping

# Prediction (requires API key)
curl -X POST https://YOUR_CLOUD_RUN_URL/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"text": "Congratulations! You have won a free prize. Call now!"}'

# Full smoke test
python src/smoke_test.py --url https://YOUR_CLOUD_RUN_URL --api-key YOUR_API_KEY
```

All 10 smoke test cases should pass.

---

## Architecture

```
GitHub Actions (push to main)
  │
  ├── Download model weights ← GCS bucket
  ├── docker build + push   → Artifact Registry
  ├── gcloud run deploy     → Cloud Run (NVIDIA L4)
  └── Prune old images      → Artifact Registry (keeps only latest)

Client → X-API-Key header → FastAPI /predict → BGE tokenizer → frozen encoder (layers 0-8)
                                                              → fine-tuned encoder (layers 9-11)
                                                              → classifier head → spam/not-spam
```
