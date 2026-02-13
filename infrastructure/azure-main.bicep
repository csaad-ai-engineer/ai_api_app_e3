// infrastructure/azure/main.bicep
// Déploiement Azure Container Apps pour AI API App E3
// Ressources : Container App Environment, API App, Streamlit App, Log Analytics

param location string = resourceGroup().location
param envName  string = 'ai-api-app-e3-env'
param acrLoginServer string
param imageTagApi       string = 'latest'
param imageTagStreamlit string = 'latest'

@secure()
param jwtSecret string
@secure()
param mlflowTrackingUri string

// ── Log Analytics Workspace ──────────────────────────────────────────────────
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${envName}-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

// ── Container App Environment ────────────────────────────────────────────────
resource containerAppEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: envName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey:  logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// ── API Container App ────────────────────────────────────────────────────────
resource apiApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'ai-api-app-api'
  location: location
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      ingress: {
        external:   true
        targetPort: 8000
        transport:  'http'
      }
      secrets: [
        { name: 'jwt-secret',            value: jwtSecret }
        { name: 'mlflow-tracking-uri',   value: mlflowTrackingUri }
      ]
    }
    template: {
      containers: [{
        name:  'ai-api-app-api'
        image: '${acrLoginServer}/ai-api-app-api:${imageTagApi}'
        resources: { cpu: '0.5', memory: '1Gi' }
        env: [
          { name: 'JWT_SECRET',           secretRef: 'jwt-secret' }
          { name: 'MLFLOW_TRACKING_URI',  secretRef: 'mlflow-tracking-uri' }
          { name: 'MODEL_NAME',  value: 'ai-api-app-model' }
          { name: 'MODEL_STAGE', value: 'Production' }
        ]
        probes: [{
          type: 'Liveness'
          httpGet: { path: '/health', port: 8000 }
          initialDelaySeconds: 30
          periodSeconds: 30
        }]
      }]
      scale: {
        minReplicas: 1
        maxReplicas: 5
        rules: [{
          name: 'http-scaling'
          http: { metadata: { concurrentRequests: '20' } }
        }]
      }
    }
  }
}

// ── Streamlit Container App ──────────────────────────────────────────────────
resource streamlitApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'ai-api-app-streamlit'
  location: location
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      ingress: {
        external:   true
        targetPort: 8501
        transport:  'http'
      }
      secrets: [
        { name: 'jwt-secret', value: jwtSecret }
      ]
    }
    template: {
      containers: [{
        name:  'ai-api-app-streamlit'
        image: '${acrLoginServer}/ai-api-app-streamlit:${imageTagStreamlit}'
        resources: { cpu: '0.25', memory: '512Mi' }
        env: [
          { name: 'API_URL',    value: 'https://${apiApp.properties.configuration.ingress.fqdn}' }
          { name: 'JWT_SECRET', secretRef: 'jwt-secret' }
        ]
      }]
      scale: { minReplicas: 1, maxReplicas: 3 }
    }
  }
}

// ── Outputs ──────────────────────────────────────────────────────────────────
output apiUrl       string = 'https://${apiApp.properties.configuration.ingress.fqdn}'
output streamlitUrl string = 'https://${streamlitApp.properties.configuration.ingress.fqdn}'
