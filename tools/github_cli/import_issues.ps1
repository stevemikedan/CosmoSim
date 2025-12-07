# $csv = Import-Csv "cosmosim_completed_issues.csv"
$csv = Import-Csv "import_backlog_issues.csv"
$repo = "stevemikedan/CosmoFrame"

foreach ($row in $csv) {

    Write-Host "Creating issue: $($row.title)"

    # Split CSV labels (comma-separated)
    $labelList = $row.labels -split "," | ForEach-Object { $_.Trim() }

    # Ensure labels exist
    $labelArgs = @()
    foreach ($label in $labelList) {
        if ($label -ne "") {
            gh label create "$label" --repo $repo --force | Out-Null
            $labelArgs += @("--label", "$label")
        }
    }

    # Create issue
    $url = gh issue create `
        --repo $repo `
        --title "$($row.title)" `
        --body "$($row.body)" `
        $labelArgs

    Write-Host " -> GitHub returned: $url"

    # Extract issue number
    if ($url -match "/issues/(\d+)$") {
        $issueNumber = $matches[1]
        Write-Host " -> Created issue #$issueNumber"
    } else {
        Write-Host " -> ERROR: Could not extract issue number"
        continue
    }

    # Close if needed
    if ($row.state -eq "closed") {
        gh issue close $issueNumber --repo $repo
    }
}
