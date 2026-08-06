{{/*
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}}

{{- define "switchyard.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "switchyard.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "switchyard.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "switchyard.labels" -}}
helm.sh/chart: {{ include "switchyard.chart" . }}
{{ include "switchyard.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: switchyard
{{- end }}

{{- define "switchyard.selectorLabels" -}}
app.kubernetes.io/name: {{ include "switchyard.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "switchyard.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "switchyard.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Name of the ConfigMap holding the deployment TOML.
*/}}
{{- define "switchyard.configMapName" -}}
{{- if .Values.config.create }}
{{- include "switchyard.fullname" . }}
{{- else }}
{{- required "config.existingConfigMap is required when config.create is false" .Values.config.existingConfigMap }}
{{- end }}
{{- end }}

{{/*
Name of the Secret providing upstream API keys, or "" when none is configured.
*/}}
{{- define "switchyard.apiKeySecretName" -}}
{{- if .Values.apiKeySecret.create }}
{{- default (include "switchyard.fullname" .) .Values.apiKeySecret.name }}
{{- else }}
{{- .Values.apiKeySecret.name }}
{{- end }}
{{- end }}

{{- define "switchyard.image" -}}
{{- printf "%s:%s" .Values.image.repository (default .Chart.AppVersion .Values.image.tag) }}
{{- end }}

{{/*
Render a probe, defaulting its scheme to HTTPS when Switchyard terminates TLS.

kubelet probes default to HTTP. Against a TLS listener that fails, so the pod
would never pass its probes when tls.enabled is set. An explicitly configured
scheme always wins.
*/}}
{{- define "switchyard.probe" -}}
{{- $probe := deepCopy .probe -}}
{{- if and .root.Values.tls.enabled (hasKey $probe "httpGet") -}}
{{- if not (hasKey $probe.httpGet "scheme") -}}
{{- $_ := set $probe.httpGet "scheme" "HTTPS" -}}
{{- end -}}
{{- end -}}
{{- toYaml $probe -}}
{{- end }}

{{/*
Absolute path to the deployment TOML inside the container.
*/}}
{{- define "switchyard.configPath" -}}
{{- printf "%s/%s" (trimSuffix "/" .Values.config.mountPath) .Values.config.key }}
{{- end }}
