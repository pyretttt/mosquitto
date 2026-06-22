{{/*
Common labels for the app CRDs this chart manages.
The `release` label is what the Prometheus Operator uses to adopt the CRD.
*/}}
{{- define "monitoring.crdLabels" -}}
release: {{ .Values.app.releaseLabel }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: mlopster
{{- end }}
