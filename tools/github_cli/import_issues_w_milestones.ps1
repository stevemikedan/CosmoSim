# $csv = Import-Csv "cosmosim_completed_issues.csv"
$csv = Import-Csv "cosmoframe_issues.csv"
$repo = "stevemikedan/CosmoFrame"

foreach ($row in $csv) {

    Write-Host "Creating issue: $($row.title)"

    # ------------------------------------------------------------
    # Split CSV labels (comma-separated)
    # ------------------------------------------------------------
    $labelList = $row.labels -split "," | ForEach-Object { $_.Trim() }

    # Ensure labels exist
    $labelArgs = @()
    foreach ($label in $labelList) {
        if ($label -ne "") {
            gh label create "$label" --repo $repo --force | Out-Null
            $labelArgs += @("--label", "$label")
        }
    }

    # ------------------------------------------------------------
    # Ensure milestone exists & fetch milestone number
    # ------------------------------------------------------------
    $milestoneNumber = $null

    if ($row.milestone -and $row.milestone.Trim() -ne "") {
        $milestoneName = $row.milestone.Trim()

        # Try to find the milestone
        $existing = gh api repos/$repo/milestones --jq ".[] | select(.title==\"$milestoneName\") | .number"

        # If milestone does not exist, create it
        if (-not $existing) {
            Write-Host " -> Creating milestone: $milestoneName"
            gh milestone create "$milestoneName" --repo $repo | Out-Null

            # Fetch its number again
            $existing = gh api repos/$repo/milestones --jq ".[] | select(.title==\"$milestoneName\") | .number"
        }

        $milestoneNumber = $existing
    }

    # ------------------------------------------------------------
    # Create issue (with milestone if present)
    # ------------------------------------------------------------
    $createArgs = @(
        "--repo", $repo,
        "--title", "$($row.title)",
        "--body", "$($row.body)"
    ) + $labelArgs

    if ($milestoneNumber) {
        $createArgs += @("--milestone", "$milestoneNumber")
    }

    $url = gh issue create @createArgs

    Write-Host " -> GitHub returned: $url"

    # ------------------------------------------------------------
    # Extract Issue Number
    # ------------------------------------------------------------
    if ($url -match "/issues/(\d+)$") {
        $issueNumber = $matches[1]
        Write-Host " -> Created issue #$issueNumber"
    } else {
        Write-Host " -> ERROR: Could not extract issue number"
        continue
    }

    # ------------------------------------------------------------
    # Close issue if CSV specifies state=closed
    # ------------------------------------------------------------
    if ($row.state -eq "closed") {
        Write-Host " -> Closing issue #$issueNumber"
        gh issue close $issueNumber --repo $repo
    }
}
