# Save this as reinstall_extensions.ps1
Write-Host "Installing VS Code extensions..." -ForegroundColor Green

Get-Content my_extensions.txt | ForEach-Object {
    Write-Host "Installing: $_" -ForegroundColor Yellow
    code --install-extension $_
}

Write-Host "All extensions installed!" -ForegroundColor Green
Write-Host "Don't forget to restore settings from vscode_backup folder" -ForegroundColor Cyan
