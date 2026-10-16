param(
    [string]$OutputDir = "$PSScriptRoot\..\models"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$models = @(
    @{
        Name = "palm_detection_mediapipe_2023feb.onnx"
        Url = "https://huggingface.co/opencv/palm_detection_mediapipe/resolve/main/palm_detection_mediapipe_2023feb.onnx"
    },
    @{
        Name = "handpose_estimation_mediapipe_2023feb.onnx"
        Url = "https://huggingface.co/opencv/handpose_estimation_mediapipe/resolve/main/handpose_estimation_mediapipe_2023feb.onnx"
    }
)

foreach ($model in $models) {
    $target = Join-Path $OutputDir $model.Name
    if (Test-Path $target) {
        Write-Host "Already exists: $target"
        continue
    }

    Write-Host "Downloading $($model.Name)..."
    Invoke-WebRequest -Uri $model.Url -OutFile $target
}

Write-Host "Models are ready in $OutputDir"
