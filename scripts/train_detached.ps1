# Launch a WSL training step detached from the current terminal.
# Usage: powershell -File scripts\train_detached.ps1 smoke --max_steps 10
param([string]$Name, [Parameter(ValueFromRemainingArguments = $true)][string[]]$TrainArgs)
$wslPath = "/mnt/c/Users/$env:USERNAME/Documents/CCP-Car-Assistant"
$cmd = "cd '$wslPath' && sed -i 's/\r$//' finetune/run_train.sh && bash finetune/run_train.sh $Name $($TrainArgs -join ' ')"
Start-Process -WindowStyle Hidden -FilePath "wsl.exe" -ArgumentList "-d", "Ubuntu", "--", "bash", "-lc", "`"$cmd`""
Write-Host "Started '$Name'. Follow with: Get-Content outputs\$Name.log -Wait -Tail 20"
