param(
    [string]$InputDocx = (Join-Path $PSScriptRoot '../reports/CLUENER2020_课程实验报告.docx'),
    [string]$OutputDir = (Join-Path $PSScriptRoot '../runs/course_report_qa')
)
$ErrorActionPreference = 'Stop'
$docxPath = (Resolve-Path -LiteralPath $InputDocx).Path
$qaPath = [System.IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Path $qaPath -Force | Out-Null
$pdfPath = Join-Path $qaPath 'course_report.pdf'
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $document = $word.Documents.Open($docxPath, $false, $true, $false)
    $document.Fields.Update() | Out-Null
    $document.Repaginate()
    $pages = $document.ComputeStatistics(2)
    $document.ExportAsFixedFormat($pdfPath, 17)
    [PSCustomObject]@{ Pdf=$pdfPath; Pages=$pages; Renderer='Microsoft Word'; SourceReadOnly=$true } | ConvertTo-Json
} finally {
    if ($null -ne $document) {
        $document.Close(0)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document)
    }
    if ($null -ne $word) {
        $word.Quit(0)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
}
