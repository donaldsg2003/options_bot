param(
    [Parameter(Mandatory=$true)]
    [string]$SearchPhrase,
    
    [string]$Path = ".",
    
    [string[]]$Include = @("*.py", "*.sql", "*.txt", "*.md", "*.sh"),
    
    [switch]$CaseSensitive
)

Write-Host "`nSearching for: '$SearchPhrase'" -ForegroundColor Cyan
Write-Host "In path: $(Resolve-Path $Path)" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Gray

$results = @()

Get-ChildItem -Path $Path -Recurse -Include $Include -File -ErrorAction SilentlyContinue | ForEach-Object {
    $file = $_
    $lineNumber = 0
    
    Get-Content $file.FullName -ErrorAction SilentlyContinue | ForEach-Object {
        $lineNumber++
        $line = $_
        
        $found = if ($CaseSensitive) {
            $line -cmatch [regex]::Escape($SearchPhrase)
        } else {
            $line -match [regex]::Escape($SearchPhrase)
        }
        
        if ($found) {
            $results += [PSCustomObject]@{
                File = $file.FullName
                Line = $lineNumber
                Content = $line.Trim()
            }
        }
    }
}

if ($results.Count -eq 0) {
    Write-Host "`nNo matches found." -ForegroundColor Yellow
} else {
    Write-Host "`nFound $($results.Count) matches:" -ForegroundColor Green
    Write-Host ("=" * 70) -ForegroundColor Gray
    
    $currentFile = ""
    foreach ($result in $results) {
        if ($result.File -ne $currentFile) {
            Write-Host "`n$($result.File)" -ForegroundColor Cyan
            $currentFile = $result.File
        }
        Write-Host "  Line $($result.Line): " -ForegroundColor Yellow -NoNewline
        Write-Host $result.Content -ForegroundColor White
    }
    
    Write-Host "`n" -NoNewline
}