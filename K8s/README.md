## How to run the kubernetes locally

```bash
minikube start
```

We will create a kubernetes deployment using the yaml file
```bash
kubectl create -f deployment.yaml
```

To check if the pods is running locally
```bash
kubectl get pods
```

Create a kubernetes service using the yaml file
```bash
kubectl create -f service.yaml
```

To check if the service is running 
```bash
kubectl get service
```

Create a ingress using the yaml file
```bash
kubectl create -f ingress.yaml
```

To check if the ingress is running 
```bash
kubectl get ingress
```