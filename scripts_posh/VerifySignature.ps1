<#
.SYNOPSIS
  Create or reuse an RSA keypair and sign all artifacts in a directory, sign the manifest, or verify existing signatures.

.PARAMETER InputDir
  Root directory to scan, sign, or verify.

.PARAMETER IncludeDirs
  Optional array of directories to include (relative to InputDir or absolute). If omitted, the entire InputDir is processed.

.PARAMETER KeyDir
  Directory (relative to InputDir or absolute) where keys are stored. Defaults to "verify_sign" under InputDir.
#>

#requires -Version 7.0
[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseApprovedVerbs', '')]
param(
    [Parameter(Mandatory=$true)]
    [string]$InputDir,
    [string[]]$IncludeDirs,
    [string[]]$IncludeFiles,
    [string]$KeyDir = "verify_sign"
)

if ($PSVersionTable.PSVersion.Major -lt 7) {
    Write-Error "PowerShell 7 or newer is required. You can run this script inside the VS Code terminal, which uses PowerShell 7 by default."
    exit 2
}

# Resolve InputDir
$InputDir = (Resolve-Path -Path $InputDir).Path.TrimEnd('\','/')

# Resolve KeyDir: if absolute use as-is, otherwise relative to InputDir
try { $KeyDir = (Resolve-Path -Path $KeyDir -ErrorAction Stop).Path.TrimEnd('\','/') } catch {
      $KeyDir = (Resolve-Path -Path (New-Item -ItemType Directory -Path $KeyDir -Force -ErrorAction SilentlyContinue).FullName).Path.TrimEnd('\','/')
}

$PublicPem  = Join-Path $KeyDir "RAG_LCC_public.pem"
$Manifest   = Join-Path $InputDir "signed.txt"
$ManifestSig = $Manifest + ".sig"   # detached signature for the manifest


function Import-PublicRsa {
    param([string]$Path)
    $pem = Get-Content -Path $Path -Raw

    $rsa = [System.Security.Cryptography.RSA]::Create()

    # Prefer ImportFromPem if available
    $importFromPemMethod = $rsa.GetType().GetMethod("ImportFromPem", [System.Reflection.BindingFlags]::Instance -bor [System.Reflection.BindingFlags]::Public)
    if ($importFromPemMethod) {
        try {
            $rsa.ImportFromPem($pem)
            return $rsa
        } catch {
            Write-Warning "ImportFromPem failed for public key, falling back to SubjectPublicKeyInfo import: $_"
        }
    }

    # Fallback: decode base64 and call ImportSubjectPublicKeyInfo via reflection to handle ReadOnlySpan<byte>
    $b64 = ($pem -split "`n" | Where-Object { $_ -and ($_ -notmatch '^-----') }) -join ''
    $bytes = [Convert]::FromBase64String($b64)

    try {
        $method = $rsa.GetType().GetMethod("ImportSubjectPublicKeyInfo", [System.Reflection.BindingFlags]::Instance -bor [System.Reflection.BindingFlags]::Public)
        if (-not $method) {
            throw "ImportSubjectPublicKeyInfo method not found on RSA type."
        }

        $rosType = [Type]::GetType("System.ReadOnlySpan`1[[System.Byte, System.Private.CoreLib]], System.Private.CoreLib")
        if ($rosType) {
            $op = $rosType.GetMethod("op_Implicit", [System.Reflection.BindingFlags]::Static -bor [System.Reflection.BindingFlags]::Public)
            if ($op) {
                $span = $op.Invoke($null, @([object]$bytes))
                $bytesRead = 0
                $method.Invoke($rsa, @($span, [ref]$bytesRead)) | Out-Null
                return $rsa
            }
        }

        $bytesRead = 0
        $method.Invoke($rsa, @($bytes, [ref]$bytesRead)) | Out-Null
        return $rsa
    } catch {
        throw "Failed to load public key: $_"
    }
}

# Normalize include dirs to full paths if provided
$targetDirs = @()
if ($IncludeDirs -and $IncludeDirs.Count -gt 0) {
    foreach ($d in $IncludeDirs) {
        if ([string]::IsNullOrWhiteSpace($d)) { continue }
        $candidate = if ([IO.Path]::IsPathRooted($d)) { $d } else { Join-Path $InputDir $d }
        try {
            $full = (Resolve-Path -Path $candidate -ErrorAction Stop).Path
            $full = $full.TrimEnd('\','/')
            $targetDirs += $full
        } catch {
            Write-Warning "Include directory not found or inaccessible: $candidate — skipping"
            continue
        }
    }
}

# Resolve IncludeFiles globs against InputDir root
$targetFiles = @()
if ($IncludeFiles -and $IncludeFiles.Count -gt 0) {
    foreach ($pattern in $IncludeFiles) {
        if ([string]::IsNullOrWhiteSpace($pattern)) { continue }
        $resolved = Get-ChildItem -Path $InputDir -File -Filter $pattern -ErrorAction SilentlyContinue
        foreach ($f in $resolved) {
            $targetFiles += $f.FullName
        }
    }
    if ($targetFiles.Count -gt 0) {
        Write-Host "Matched $($targetFiles.Count) root-level file(s) from IncludeFiles globs."
    }
}

function Test-InTargetDirs {
    param([string]$FullPath)
    if ($targetDirs.Count -eq 0 -and $targetFiles.Count -eq 0) { return $true }
    foreach ($td in $targetDirs) {
        if ($FullPath.StartsWith($td, [System.StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    foreach ($tf in $targetFiles) {
        if ($FullPath -eq $tf) { return $true }
    }
    return $false
}

if (-not (Test-Path $PublicPem)) { Write-Error "Public key not found at $PublicPem. Cannot verify."; exit 4 }
try { $rsaPub = Import-PublicRsa -Path $PublicPem } catch { Write-Error "Failed to load public key: $_"; exit 5 }

$pairs = @()

# If manifest exists, prefer it but verify its signature first if present
if (Test-Path $Manifest) {
    $manifestTrusted = $false
    if (Test-Path $ManifestSig) {
        try {
            $manifestBytes = [System.IO.File]::ReadAllBytes($Manifest)
            $manifestSigB64 = (Get-Content -Path $ManifestSig -Raw).Trim()
            $manifestSigBytes = [Convert]::FromBase64String($manifestSigB64)
            $manifestOk = $rsaPub.VerifyData($manifestBytes, $manifestSigBytes, [System.Security.Cryptography.HashAlgorithmName]::SHA256, [System.Security.Cryptography.RSASignaturePadding]::Pkcs1)
            if ($manifestOk) {
                Write-Host "Manifest signature valid. Using manifest entries."
                $manifestTrusted = $true
            } else {
                Write-Warning "Manifest signature INVALID. Will not trust manifest; falling back to .sig files."
            }
        } catch {
            Write-Warning "Error verifying manifest signature: $_. Will not trust manifest; falling back to .sig files."
        }
    } else {
        Write-Warning "Manifest signature file not found ($ManifestSig). Manifest is not trusted; falling back to .sig files."
    }

    if ($manifestTrusted) {
        $lines = Get-Content -Path $Manifest
        foreach ($line in $lines) {
            if (-not $line) { continue }
            $parts = $line -split '\|',2
            if ($parts.Count -ne 2) { Write-Warning "Skipping malformed manifest line: $line"; continue }
            $rel = $parts[0]
            $sigB64 = $parts[1]
            $full = Join-Path $InputDir $rel
            $resolved = Resolve-Path -Path $full -ErrorAction SilentlyContinue
            $full = if ($resolved) { $resolved.Path } else { $null }
            if (-not $full) { Write-Warning "Referenced file not found: $rel"; continue }
            if (-not (Test-InTargetDirs -FullPath $full)) { continue }
            $pairs += [PSCustomObject]@{ Relative = $rel; File = $full; Sig = $sigB64 }
        }
        if ($pairs.Count -eq 0) { Write-Error "No manifest entries matched the selected directories."; exit 6 }
    }
}

# If we don't have trusted manifest entries, attempt to find .sig files on disk
if ($pairs.Count -eq 0) {
    Write-Host "Attempting verification using .sig files on disk."
    $sigFiles = Get-ChildItem -Path $InputDir -Recurse -File -Filter "*.sig" | Where-Object { $_.FullName -notlike "*\signing-key\*" -and $_.FullName -ne $ManifestSig }
    if ($targetDirs.Count -gt 0) {
        $sigFiles = $sigFiles | Where-Object {
            $orig = $_.FullName.Substring(0, $_.FullName.Length - 4)
            Test-InTargetDirs -FullPath $orig
        }
    }
    if ($sigFiles.Count -eq 0) { Write-Error "No manifest and no .sig files found to verify in the selected directories."; exit 6 }
    foreach ($s in $sigFiles) {
        $orig = $s.FullName.Substring(0, $s.FullName.Length - 4)
        $rel = $orig.Substring($InputDir.Length).TrimStart('\','/')
        $sigB64 = Get-Content -Path $s.FullName -Raw
        $pairs += [PSCustomObject]@{ Relative = $rel; File = $orig; Sig = $sigB64 }
    }
}

$failures = @()
foreach ($p in $pairs) {
    try {
        if (-not (Test-Path $p.File)) { Write-Warning "File missing: $($p.File)"; $failures += $p; continue }
        $bytes = [System.IO.File]::ReadAllBytes($p.File)
        $sigBytes = [Convert]::FromBase64String($p.Sig.Trim())
        $ok = $rsaPub.VerifyData($bytes, $sigBytes, [System.Security.Cryptography.HashAlgorithmName]::SHA256, [System.Security.Cryptography.RSASignaturePadding]::Pkcs1)
        if ($ok) {
            Write-Host "OK: $($p.Relative)"
        } else {
            Write-Warning "INVALID: $($p.Relative)"
            $failures += $p
        }
    } catch {
        Write-Warning "Error verifying $($p.Relative): $_"
        $failures += $p
    }
}

if ($failures.Count -eq 0) {
    Write-Host "All signatures valid."
    exit 0
} else {
    Write-Error ("Verification failed for {0} item(s)." -f $failures.Count)
    exit 7
}
