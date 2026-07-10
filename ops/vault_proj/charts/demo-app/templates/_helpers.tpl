{{/*
demo-app helpers — intentionally minimal for the lab.
*/}}
{{- define "demo-app.name" -}}
{{- .Values.name | default .Chart.Name -}}
{{- end -}}
