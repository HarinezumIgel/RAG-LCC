<#
.SYNOPSIS
  Gather third-party licenses from a venv, output JSON + Markdown + SPDX,
  wrapping and stripping venv paths.

.DESCRIPTION
  • Requires: ProjectPath, VenvPath, LicenseDir
  • Builds pythonExe from VenvPath
  • Invokes pip-licenses as a module (python -m piplicenses)
  • Emits THIRD_PARTY_LICENSES.json/.md/.spdx under RequirementsAndLicenses
#>

[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseApprovedVerbs', '')]
param(
  [Parameter(Mandatory = $true, HelpMessage = 'Full path to your Python project directory')]
  [Alias('input-dir')]
  [ValidateNotNullOrEmpty()]
  [string]$ProjectPath,

  [Parameter(Mandatory = $true, HelpMessage = 'Full path to your virtualenv root (folder containing Scripts/)')]
  [ValidateNotNullOrEmpty()]
  [string]$VenvName,

  [Parameter(Mandatory = $true, HelpMessage = 'Directory containing the output')]
  [ValidateNotNullOrEmpty()]
  [string]$ContainingLicenseDirectoryName
)

function wrap {
  param(
    [string]$Text,
    [int]   $Width = 80
  )
  $words   = $Text -split '\s+'
  $lines   = [System.Collections.Generic.List[string]]::new()
  $current = ''

  foreach ($w in $words) {
    if (($current.Length + $w.Length + 1) -le $Width) {
      $current = if ($current) { "$current $w" } else { $w }
    }
    else {
      if ($current) { $lines.Add($current) }
      $current = $w
    }
  }
  if ($current) { $lines.Add($current) }

  return ,($lines.ToArray())
}

function Get-SHA256Hex {
  param([string]$Path)
  if (-not (Test-Path $Path)) { throw "File not found: $Path" }
  $bytes = [System.IO.File]::ReadAllBytes($Path)
  $sha256 = [System.Security.Cryptography.SHA256]::Create()
  $hashBytes = $sha256.ComputeHash($bytes)
  $hashHex = ($hashBytes | ForEach-Object { $_.ToString("x2") }) -join ''
  return $hashHex
}

function Write-SHA256File {
  param(
    [string]$FilePath
  )
  $shaHex = Get-SHA256Hex -Path $FilePath
  $shaFile = "$FilePath.sha256"
  Set-Content -Path $shaFile -Value $shaHex -Encoding UTF8
  return $shaFile, $shaHex
}

# 1) Validate project root
if (-not (Test-Path $ProjectPath -PathType Container)) {
  Write-Error "'$ProjectPath' is not a valid folder."
  exit 1
}

# 2) Prepare output folder
$LicenseDir = Join-Path $ProjectPath $ContainingLicenseDirectoryName
if (-not (Test-Path $LicenseDir)) {
  New-Item -Path $LicenseDir -ItemType Directory -Force | Out-Null
}

$LicenseDirNameOnly = (Split-Path $LicenseDir -Leaf) -replace '\.[^.]+$',''
$jsonFile = Join-Path $LicenseDir 'Licenses.json'
$mdFile   = Join-Path $LicenseDir 'Licenses.md'
$spdxFile = Join-Path $LicenseDir 'Licenses.spdx'

# If caller provided VenvNameForFile earlier in environment, use it; otherwise derive safe suffix
if ($env:VenvNameForFile) { $VenvNameForFile = $env:VenvNameForFile } else { $VenvNameForFile = '' }

# 3) Ensure UTF-8 python
$env:PYTHONUTF8 = '1'

# 4) Build path to python.exe in the given venv
$VenvPath = Join-Path $ProjectPath $VenvName
$pythonExe = Join-Path $VenvPath 'Scripts\python.exe'
if (-not (Test-Path $pythonExe)) {
  Write-Error "python.exe not found in '$VenvPath\Scripts'."
  exit 1
}


# 6) Invoke pip-licenses via the venv’s python interpreter
Write-Host "`nGenerating JSON license report via pip-licenses..." -ForegroundColor Cyan
Write-Host " & $pythonExe -m piplicenses  --with-system  --with-license-file   --with-authors   --with-urls   --format json"
$jsonOutput = & $pythonExe -m piplicenses `
  --with-system `
  --with-license-file `
  --with-authors `
  --with-urls `
  --format json

if ($LASTEXITCODE -ne 0) {
  Write-Error 'pip-licenses failed.'
  exit $LASTEXITCODE
}

# 7) Prepare disclaimer text for embedding
$disclaimer = @"
This overview is provided for informational purposes only. Signing attests
only to the integrity of this document, not to the legal validity,
authorship, or completeness of any third‑party licenses referenced herein.
Users are responsible for reviewing and complying with all third‑party terms.
"@

# 8) Parse JSON, then sanitize venv paths in LicenseFile fields
try {
  $rawPackages = $jsonOutput | ConvertFrom-Json
} catch {
  Write-Error "Failed to parse pip-licenses JSON output: $_"
  exit 1
}

# Replace the absolute venv path prefix with a portable placeholder.
# Match everything up to and including Lib\site-packages\
$venvPathNorm = $VenvPath.TrimEnd('\','/')
foreach ($pkg in $rawPackages) {
  if ($pkg.LicenseFile -and $pkg.LicenseFile -ne 'UNKNOWN') {
    $pkg.LicenseFile = $pkg.LicenseFile -replace [regex]::Escape($venvPathNorm), '<your-venv>'
  }
}

# 9) Write JSON with disclaimer as top-level metadata
$jsonObject = [PSCustomObject]@{
    disclaimer = $disclaimer.Trim()
    generated  = (Get-Date).ToString("o")
    packages   = $rawPackages
}

$jsonObject |
  ConvertTo-Json -Depth 10 |
  Out-File -FilePath $jsonFile -Encoding utf8

Write-Host "JSON report: $jsonFile`n" -ForegroundColor Green

# 10) Load JSON and assemble license data (use $rawPackages already parsed)
$packages = foreach ($pkg in $rawPackages) {
  $text = ''
  if ($pkg.LicenseDir -and (Test-Path $pkg.LicenseDir)) {
    try { $text = Get-Content $pkg.LicenseDir -Raw } catch { $text = "[Error reading $($pkg.LicenseDir)]" }
  }
  [PSCustomObject]@{
    Name        = $pkg.Name
    Version     = $pkg.Version
    License     = $pkg.License
    Author      = $pkg.Author
    LicenseText = $text.Trim()
    URL         = $pkg.Url
  }
}

# 11) Build summary table
$summaryLines = @(
  "| Name                                     | Version                | License                                                        | Author                                          |",
  "|------------------------------------------|------------------------|----------------------------------------------------------------|-------------------------------------------------|"
)
foreach ($p in $packages) {
  $authorLines = wrap -Text ($p.Author -replace '\s+', ' ') -Width 30
  $n = $p.Name.PadRight(40)
  $v = $p.Version.PadRight(22)
  $l = $p.License.PadRight(62)

  $summaryLines += "| $n | $v | $l |$($authorLines[0].PadRight(48)) |"
  for ($i = 1; $i -lt $authorLines.Count; $i++) {
    $summaryLines += "|".PadRight(43) + "|" + "".PadRight(24) + "|" + "".PadRight(64) + "|$($authorLines[$i].PadRight(48)) |"
  }
}

# 12) Build detailed license sections with Markdown disclaimer header
$detailLines = @(
  "# License Overview Disclaimer",
  "",
  "This overview is provided for informational purposes only. Signing attests only to the integrity of this document, not to the legal validity, authorship, or completeness of any third‑party licenses referenced herein.",
  "",
  "Users are responsible for reviewing and complying with all third‑party terms.",
  "",
  "---",
  "",
  "# Third-Party Licenses",
  "",
  "## Summary"
)

$detailLines += $summaryLines
$detailLines += "", "## Full License Texts"
foreach ($p in $packages) {
  $detailLines += "", "### $($p.Name) $($p.Version)", "**License:** $($p.License)"

  $label       = '**Author:** '
  $wrappedAuth = wrap -Text $p.Author -Width 60
  $detailLines += "$label$($wrappedAuth[0])"
  for ($i = 1; $i -lt $wrappedAuth.Count; $i++) {
    $detailLines += (' ' * $label.Length) + $wrappedAuth[$i]
  }

  $detailLines += "", '```text'
  $detailLines += $p.LicenseText
  $detailLines += '```'
}

# 13) Write Markdown
$detailLines | Set-Content -Path $mdFile -Encoding utf8
Write-Host "`nMarkdown report: $mdFile" -ForegroundColor Green

# 14) Compute and write SHA256 checksums for JSON and MD
try {
  $jsonShaPath, $jsonSha = Write-SHA256File -FilePath $jsonFile
  Write-Host "🔒 JSON checksum written: $jsonShaPath" -ForegroundColor Green
} catch {
  Write-Warning "Failed to write JSON checksum: $_"
}

try {
  $mdShaPath, $mdSha = Write-SHA256File -FilePath $mdFile
  Write-Host "🔒 Markdown checksum written: $mdShaPath" -ForegroundColor Green
} catch {
  Write-Warning "Failed to write Markdown checksum: $_"
}

# 15) Emit a minimal SPDX tag-value file (best-effort) with disclaimer comments
try {
  $docName = "$($LicenseDirNameOnly)$VenvNameForFile"
  $spdxLines = @(
    "# SPDX Document Disclaimer",
    "# This overview is provided for informational purposes only.",
    "# Signing attests only to the integrity of this document, not to the",
    "# legal validity, authorship, or completeness of any third‑party licenses.",
    "# Users are responsible for reviewing and complying with all third‑party terms.",
    ""
  )

  $spdxLines += "SPDXVersion: SPDX-2.2"
  $spdxLines += "DataLicense: CC0-1.0"
  $spdxLines += "SPDXID: SPDXRef-DOCUMENT"
  $spdxLines += "DocumentName: $docName"
  $spdxLines += "DocumentNamespace: http://example.org/spdxdocs/$docName-$(Get-Date -Format 'yyyyMMddHHmmss')"
  $spdxLines += "Creator: pip-licenses"
  $spdxLines += "Created: $(Get-Date).ToUniversalTime().ToString('o')"
  $spdxLines += ""

  foreach ($p in $packages) {
    $pkgId = "SPDXRef-Package-$($p.Name -replace '[^A-Za-z0-9\.-]','-')"
    $spdxLines += "#####"
    $spdxLines += "PackageName: $($p.Name)"
    $spdxLines += "SPDXID: $pkgId"
    $spdxLines += "PackageVersion: $($p.Version)"
    $spdxLines += "PackageDownloadLocation: NOASSERTION"
    $spdxLines += "FilesAnalyzed: false"
    $licenseVal = if ($p.License -and $p.License -ne '') { $p.License } else { "NOASSERTION" }
    $spdxLines += "PackageLicenseConcluded: $licenseVal"
    $spdxLines += "PackageLicenseDeclared: $licenseVal"
    if ($p.URL) { $spdxLines += "PackageHomePage: $($p.URL)" }
    $spdxLines += ""
  }

  $spdxLines | Set-Content -Path $spdxFile -Encoding utf8
  Write-Host "SPDX tag-value file written: $spdxFile" -ForegroundColor Green

  # compute SPDX checksum
  $spdxShaPath, $spdxSha = Write-SHA256File -FilePath $spdxFile
  Write-Host "🔒 SPDX checksum written: $spdxShaPath" -ForegroundColor Green
} catch {
  Write-Warning "Failed to generate SPDX or checksum: $_"
}

# 16) Final summary
Write-Host "`nArtifacts generated:" -ForegroundColor Cyan
Write-Host " - JSON : $jsonFile"
if ($jsonSha) { Write-Host "   checksum: $jsonSha" -ForegroundColor DarkGray }
Write-Host " - Markdown : $mdFile"
if ($mdSha) { Write-Host "   checksum: $mdSha" -ForegroundColor DarkGray }
Write-Host " - SPDX : $spdxFile"
if ($spdxSha) { Write-Host "   checksum: $spdxSha" -ForegroundColor DarkGray }

Write-Host "`nDone." -ForegroundColor Green

