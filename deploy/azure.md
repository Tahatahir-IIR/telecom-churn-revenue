# Deploy to Azure

Target: the API on Azure Container Apps (consumption plan, scales to zero) and the scored tables on Azure Database for PostgreSQL Flexible Server (Burstable B1ms). Both are covered by the Azure free account credits, and the container app costs nothing while idle.

Everything below is the Azure CLI. Replace `<unique>` with something like your initials plus a number.

## 1. Resource group and registry

```bash
az login
az group create --name churn-rg --location francecentral

az acr create --resource-group churn-rg --name churnacr<unique> --sku Basic
az acr login --name churnacr<unique>
```

## 2. Build and push the image

The image contains the trained model, so train first.

```bash
PYTHONPATH=src python -m churn.train
docker build -t churnacr<unique>.azurecr.io/churn-api:1.0.0 .
docker push churnacr<unique>.azurecr.io/churn-api:1.0.0
```

## 3. PostgreSQL

```bash
az postgres flexible-server create \
  --resource-group churn-rg --name churn-pg-<unique> \
  --location francecentral --tier Burstable --sku-name Standard_B1ms \
  --storage-size 32 --version 16 \
  --admin-user churn --admin-password '<strong-password>' \
  --public-access 0.0.0.0-255.255.255.255

az postgres flexible-server db create \
  --resource-group churn-rg --server-name churn-pg-<unique> --database-name churn
```

Load the tables and views from your machine:

```bash
export DATABASE_URL='postgresql+psycopg2://churn:<strong-password>@churn-pg-<unique>.postgres.database.azure.com:5432/churn?sslmode=require'
PYTHONPATH=src python -m churn.db
```

Power BI connects to the same host. Restrict `--public-access` to your IP afterwards.

## 4. Container App

```bash
az containerapp env create --resource-group churn-rg --name churn-env --location francecentral

az containerapp create \
  --resource-group churn-rg --name churn-api --environment churn-env \
  --image churnacr<unique>.azurecr.io/churn-api:1.0.0 \
  --registry-server churnacr<unique>.azurecr.io \
  --target-port 8000 --ingress external \
  --cpu 0.5 --memory 1.0Gi --min-replicas 0 --max-replicas 2

az containerapp show --resource-group churn-rg --name churn-api --query properties.configuration.ingress.fqdn -o tsv
```

Open `https://<fqdn>/docs`. The first request after idle takes a few seconds while the replica starts.

## 5. Redeploy after retraining

```bash
PYTHONPATH=src python -m churn.train
docker build -t churnacr<unique>.azurecr.io/churn-api:1.0.1 .
docker push churnacr<unique>.azurecr.io/churn-api:1.0.1
az containerapp update --resource-group churn-rg --name churn-api --image churnacr<unique>.azurecr.io/churn-api:1.0.1
PYTHONPATH=src python -m churn.db
```

## 6. Tear down

```bash
az group delete --name churn-rg --yes --no-wait
```

## AWS equivalent

Same image on AWS App Runner (or ECS Fargate) and Amazon RDS for PostgreSQL. The only code change is the `DATABASE_URL`.
