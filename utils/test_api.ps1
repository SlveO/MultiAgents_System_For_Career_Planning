param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [string]$SessionId = "demo-1",
    [bool]$UseRawUtf8 = $true
)

$ErrorActionPreference = "Stop"

function Invoke-JsonRequest {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("GET", "POST")] [string]$Method,
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter()][string]$JsonBody = "",
        [Parameter()][bool]$UseRaw = $true
    )

    if (-not $UseRaw) {
        if ($Method -eq "GET") {
            return Invoke-RestMethod -Method Get -Uri $Uri
        }
        return Invoke-RestMethod -Method Post -Uri $Uri -ContentType "application/json; charset=utf-8" -Body $JsonBody
    }

    $headers = @{
        "Accept" = "application/json; charset=utf-8"
    }

    if ($Method -eq "GET") {
        $resp = Invoke-WebRequest -Method Get -Uri $Uri -Headers $headers
    }
    else {
        $headers["Content-Type"] = "application/json; charset=utf-8"
        $resp = Invoke-WebRequest -Method Post -Uri $Uri -Headers $headers -Body $JsonBody
    }

    $ms = New-Object System.IO.MemoryStream
    $resp.RawContentStream.CopyTo($ms)
    $bytes = $ms.ToArray()
    $text = [System.Text.Encoding]::UTF8.GetString($bytes)

    return $text | ConvertFrom-Json
}

Write-Host "== 1) healthz =="
$health = Invoke-JsonRequest -Method "GET" -Uri "$BaseUrl/healthz" -UseRaw $UseRawUtf8
$health | ConvertTo-Json -Depth 6

Write-Host "`n== 2) assist =="
$assistBody = @{
    session_id  = $SessionId
    user_goal   = "I want to transition to data analysis in 6 months"
    text_input  = "I know Python and SQL, but my project experience is weak"
    stream      = $false
    debug_trace = $false
} | ConvertTo-Json -Depth 10

$assist = Invoke-JsonRequest -Method "POST" -Uri "$BaseUrl/v1/assist" -JsonBody $assistBody -UseRaw $UseRawUtf8
$assist | ConvertTo-Json -Depth 20

Write-Host "`n== 3) session =="
$session = Invoke-JsonRequest -Method "GET" -Uri "$BaseUrl/v1/session/$SessionId" -UseRaw $UseRawUtf8
$session | ConvertTo-Json -Depth 20

Write-Host "`n== 4) feedback =="
$feedbackBody = @{
    session_id = $SessionId
    feedback   = "Helpful advice"
    rating     = 5
} | ConvertTo-Json -Depth 10

$feedback = Invoke-JsonRequest -Method "POST" -Uri "$BaseUrl/v1/feedback" -JsonBody $feedbackBody -UseRaw $UseRawUtf8
$feedback | ConvertTo-Json -Depth 6

Write-Host "`nAll done."
