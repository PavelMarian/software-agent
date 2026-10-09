from __future__ import annotations

from software_bench.mcp.profiles._shared import (
    _arguments,
    _path,
    _schema,
)

def build() -> dict[str, Any]:
    namespace = {"type": "string", "minLength": 1}
    resource = {"type": "string", "minLength": 1}
    return {
        "name": "software-bench-kubernetes",
        "version": "1.0.0",
        "description": "Compact Kubernetes inspection, rollout, and remediation profile.",
        "tools": [
            {
                "name": "kubernetes_get",
                "description": "Get Kubernetes resources with a chosen structured output format.",
                "input_schema": _schema(
                    {
                        "resource": resource, "namespace": namespace,
                        "output": {"type": "string", "enum": ["json", "yaml", "wide"]},
                    },
                    ["resource", "namespace", "output"],
                ),
                "command": ["kubectl", "get", "{resource}", "-n", "{namespace}", "-o", "{output}"],
            },
            {
                "name": "kubernetes_describe",
                "description": "Describe a resource and its recent events.",
                "input_schema": _schema(
                    {"resource": resource, "name": resource, "namespace": namespace},
                    ["resource", "name", "namespace"],
                ),
                "command": ["kubectl", "describe", "{resource}", "{name}", "-n", "{namespace}"],
            },
            {
                "name": "kubernetes_logs",
                "description": "Read bounded recent logs from a pod or workload.",
                "input_schema": _schema(
                    {
                        "target": resource, "namespace": namespace,
                        "tail": {"type": "integer"},
                    },
                    ["target", "namespace", "tail"],
                ),
                "command": ["kubectl", "logs", "{target}", "-n", "{namespace}", "--tail", "{tail}"],
            },
            {
                "name": "kubernetes_apply",
                "description": "Apply one reviewed workspace manifest.",
                "input_schema": _schema({"manifest": _path("Kubernetes manifest.")}, ["manifest"]),
                "command": ["kubectl", "apply", "-f", "{manifest}"],
                "timeout_seconds": 300, "mutating": True,
            },
            {
                "name": "kubernetes_rollout_status",
                "description": "Wait for and report workload rollout status.",
                "input_schema": _schema(
                    {"resource": resource, "namespace": namespace}, ["resource", "namespace"]
                ),
                "command": ["kubectl", "rollout", "status", "{resource}", "-n", "{namespace}"],
                "timeout_seconds": 600,
            },
            {
                "name": "kubernetes_rollout_restart",
                "description": "Restart one named workload through the rollout API.",
                "input_schema": _schema(
                    {"resource": resource, "namespace": namespace}, ["resource", "namespace"]
                ),
                "command": ["kubectl", "rollout", "restart", "{resource}", "-n", "{namespace}"],
                "timeout_seconds": 300, "mutating": True,
            },
            {
                "name": "kubernetes_events",
                "description": "List namespace events in chronological diagnostic order.",
                "input_schema": _schema({"namespace": namespace}, ["namespace"]),
                "command": [
                    "kubectl", "get", "events", "-n", "{namespace}",
                    "--sort-by=.lastTimestamp",
                ],
            },
            {
                "name": "kubernetes_top_pods",
                "description": "Inspect current pod CPU and memory consumption.",
                "input_schema": _schema({"namespace": namespace}, ["namespace"]),
                "command": ["kubectl", "top", "pods", "-n", "{namespace}"],
            },
            {
                "name": "kubernetes_diff",
                "description": "Compare a workspace manifest with live cluster state.",
                "input_schema": _schema({"manifest": _path("Kubernetes manifest.")}, ["manifest"]),
                "command": ["kubectl", "diff", "-f", "{manifest}"],
                "timeout_seconds": 300,
            },
            {
                "name": "kubernetes_wait",
                "description": "Wait for a resource to satisfy an explicit condition.",
                "input_schema": _schema(
                    {
                        "resource": resource, "namespace": namespace,
                        "condition": {"type": "string", "minLength": 1},
                        "timeout": {"type": "string", "minLength": 1},
                    },
                    ["resource", "namespace", "condition", "timeout"],
                ),
                "command": [
                    "kubectl", "wait", "{resource}", "-n", "{namespace}",
                    "--for", "{condition}", "--timeout", "{timeout}",
                ],
                "timeout_seconds": 1800,
            },
            {
                "name": "kubernetes_scale",
                "description": "Set the desired replica count for a scalable workload.",
                "input_schema": _schema(
                    {
                        "resource": resource, "namespace": namespace,
                        "replicas": {"type": "integer"},
                    },
                    ["resource", "namespace", "replicas"],
                ),
                "command": [
                    "kubectl", "scale", "{resource}", "-n", "{namespace}",
                    "--replicas", "{replicas}",
                ],
                "timeout_seconds": 300, "mutating": True,
            },
            {
                "name": "kubernetes_delete_manifest",
                "description": "Delete only resources declared in one reviewed workspace manifest.",
                "input_schema": _schema({"manifest": _path("Kubernetes manifest.")}, ["manifest"]),
                "command": ["kubectl", "delete", "-f", "{manifest}"],
                "timeout_seconds": 300, "mutating": True,
            },
            {
                "name": "helm_template",
                "description": "Render a Helm chart locally without changing the cluster.",
                "input_schema": _schema(
                    {
                        "release": resource, "chart": _path("Workspace-relative Helm chart."),
                        "namespace": namespace, "arguments": _arguments(),
                    },
                    ["release", "chart", "namespace", "arguments"],
                ),
                "command": [
                    "helm", "template", "{release}", "{chart}", "-n", "{namespace}",
                    "{arguments...}",
                ],
                "timeout_seconds": 300,
            },
        ],
    }
