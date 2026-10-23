# Renders the deck with the locally installed Microsoft PowerPoint (COM): preview PDF + one PNG per slide.
param([string]$Deck = "$PSScriptRoot\Group41_CNN-MAE_Final_Presentation.pptx")
$out = Join-Path $PSScriptRoot "preview"
New-Item -ItemType Directory -Force $out | Out-Null
Get-ChildItem $out -Filter *.png -ErrorAction SilentlyContinue | Remove-Item -Force
$app = New-Object -ComObject PowerPoint.Application
try {
    $p = $app.Presentations.Open($Deck, $true, $false, $false)   # read-only, untitled, no window
    $p.SaveAs((Join-Path $PSScriptRoot "Group41_CNN-MAE_Final_Presentation_preview.pdf"), 32)   # 32 = ppSaveAsPDF
    for ($i = 1; $i -le $p.Slides.Count; $i++) { $p.Slides.Item($i).Export((Join-Path $out ("slide{0:D2}.png" -f $i)), "PNG", 1600, 900) }
    "rendered $($p.Slides.Count) slides"
    $p.Close()
} finally { $app.Quit(); [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null }
