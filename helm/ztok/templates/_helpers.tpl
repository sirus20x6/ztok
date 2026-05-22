{{/*
ztok helm chart — template helpers.

Follows the Bitnami common-chart conventions for naming, labels, and
selectors so users coming from any well-maintained chart will recognise
the patterns:
  * <chart>.name             — chart name, override via nameOverride
  * <chart>.fullname         — release-qualified name, override via
                               fullnameOverride
  * <chart>.chart            — chart-version label value
  * <chart>.labels           — standard label set (helm.sh + app.k8s.io)
  * <chart>.selectorLabels   — subset used in matchLabels/selectors
                               (these MUST be immutable for the lifetime
                               of the workload)
  * <chart>.serviceAccountName
  * <chart>.imagePullSecrets

Keep this file aligned with whatever helm/charts upstream is shipping
for "common"; deviations earn surprise.
*/}}

{{/*
Expand the chart name. Truncated to 63 chars because k8s label values
have a 63-char limit (DNS-1123).
*/}}
{{- define "ztok.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name. If release name contains
the chart name we strip the duplicate (Bitnami convention). Truncated
to 63 chars.
*/}}
{{- define "ztok.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Chart label value, in the form `<name>-<version>` with `+` replaced by
`_` because `+` is not a valid label value character.
*/}}
{{- define "ztok.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels — applied to every object in the chart. Includes the
upstream Kubernetes recommended labels. `app.kubernetes.io/version` is
left mutable (changes between chart releases when appVersion bumps).
*/}}
{{- define "ztok.labels" -}}
helm.sh/chart: {{ include "ztok.chart" . }}
{{ include "ztok.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: ztok
{{- end -}}

{{/*
Selector labels — the immutable subset used in `Deployment.spec.selector`
and `Service.spec.selector`. NEVER include version/release labels here:
changing them is a breaking, non-upgradable change on Deployments.
*/}}
{{- define "ztok.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ztok.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
ServiceAccount name to use.
*/}}
{{- define "ztok.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "ztok.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
Render imagePullSecrets only when at least one entry is set. Avoids
the empty `imagePullSecrets: []` field which some admission policies
flag as misconfiguration.
*/}}
{{- define "ztok.imagePullSecrets" -}}
{{- if .Values.imagePullSecrets -}}
imagePullSecrets:
{{ toYaml .Values.imagePullSecrets | indent 2 }}
{{- end -}}
{{- end -}}

{{/*
Auth secret name — points to the user-managed secret OR to the
chart-created one (when auth.create=true). Returning a single name
keeps the deployment template free of branching.
*/}}
{{- define "ztok.authSecretName" -}}
{{- if and .Values.auth.enabled .Values.auth.create -}}
{{- printf "%s-auth" (include "ztok.fullname" .) -}}
{{- else -}}
{{- .Values.auth.tokenSecretName -}}
{{- end -}}
{{- end -}}

{{/*
Full container image reference. Splits the repository@digest case from
the tagged case so users pinning by digest don't end up with
`repo@sha256:...:1.22.0`.
*/}}
{{- define "ztok.image" -}}
{{- $repo := .Values.image.repository -}}
{{- $tag  := .Values.image.tag | default .Chart.AppVersion -}}
{{- if hasPrefix "sha256:" $tag -}}
{{- printf "%s@%s" $repo $tag -}}
{{- else -}}
{{- printf "%s:%s" $repo $tag -}}
{{- end -}}
{{- end -}}
