# Replace with your actual API key
$API_KEY = $env:OpenAIApiKey 

# Initialize list length to 100 to start the loop
$listlen = 100

# Continue looping while list length is greater than or equal to 100
while ($listlen -ge 100) {
    Write-Host "Listing all assistants..."

    # Fetch the list of assistants (maximum 100 at a time)
    $assistants = Invoke-RestMethod -Uri "https://api.openai.com/v1/assistants?limit=100" `
                                    -Headers @{
                                        "Authorization" = "Bearer $API_KEY"
                                        "Content-Type" = "application/json"
                                        "OpenAI-Beta" = "assistants=v1"
                                    } -Method Get

    # Extract IDs of all assistants
    $ids = $assistants.data | ForEach-Object { $_.id }

    # Determine the number of assistants returned
    $listlen = $ids.Count

    # If no assistants are found, exit the loop
    if ($listlen -eq 0) {
        Write-Host "No assistants to delete."
        break
    }

    # Loop through each assistant ID and delete it
    foreach ($id in $ids) {
        Write-Host "Deleting assistant with ID: $id"
        $delete_response = Invoke-RestMethod -Uri "https://api.openai.com/v1/assistants/$id" `
                                             -Headers @{
                                                 "Authorization" = "Bearer $API_KEY"
                                                 "Content-Type" = "application/json"
                                                 "OpenAI-Beta" = "assistants=v1"
                                             } -Method Delete
        Write-Host $delete_response
    }
}

Write-Host "All assistants have been deleted."

# Clear the API key from memory
Remove-Variable -Name API_KEY
