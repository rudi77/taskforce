using Taskforce.Abstractions;
using OpenAI.Assistants;
using System.ClientModel;
using OpenAI.Files;
using OpenAI;
using System.Diagnostics;

namespace Taskforce.LLM
{
    public class OpenAIAssistantClient : ILLM
    {
#pragma warning disable OPENAI001        
        private readonly OpenAIClient _openAIClient;

        public OpenAIAssistantClient(string apiKey)
        {
            _openAIClient = new OpenAIClient(apiKey);
            
        }

        public OpenAIAssistantClient()
        {
            // read OpenAIApiKey from environment variable
            var apiKey = Environment.GetEnvironmentVariable("OpenAIApiKey");

            if (string.IsNullOrEmpty(apiKey))
            {
                throw new InvalidOperationException("OpenAIApiKey not set as environment variable");
            }

            _openAIClient = new OpenAIClient(apiKey);
        }

        #region ILLM interface methods

        public async Task<object?> SendMessageAsync(string systemPrompt, string userPrompt)
        {
            var assistantClient = _openAIClient.GetAssistantClient();
            var assistant = await assistantClient.CreateAssistantAsync(
                "gpt-4o-mini",
                new AssistantCreationOptions()
                {
                    Instructions = systemPrompt,
                });

            var threadOptions = new ThreadCreationOptions
            {
                InitialMessages = { userPrompt }
            };

            ThreadRun run = await assistantClient.CreateThreadAndRunAsync(assistant, threadOptions, new RunCreationOptions { Temperature = 0.0f });

            
            while (!run.Status.IsTerminal)
            {
                await Task.Delay(TimeSpan.FromSeconds(1));
                run = await assistantClient.GetRunAsync(run.ThreadId, run.Id);
            }

            if (run.Status == RunStatus.Completed)
            {
                var messagePages = assistantClient.GetMessagesAsync(run.ThreadId);

                var messages = messagePages.AsPages();
                
                await foreach (ResultPage<ThreadMessage> resultPage in messages) 
                { 
                    foreach (var message in resultPage)
                    {
                        // TODO: check content length ==  and use FirstOrDefault 
                        return message.Content.First().Text;
                    }
                      
                }
            }

            return null;
        }

        public async Task<object?> SendMessageAsync(string systemPrompt, string userPrompt, IList<string> filePaths)
        {
            var assistantClient = _openAIClient.GetAssistantClient();
            var assistant = await assistantClient.CreateAssistantAsync("gpt-4o-mini", new AssistantCreationOptions { Instructions = systemPrompt });
            var uploadedFiles = await UploadFilesForVisionAsync(filePaths);
            var messageContentList = string.IsNullOrEmpty(userPrompt) 
                ? new List<MessageContent>() 
                : new List<MessageContent> { userPrompt };

            uploadedFiles.ForEach(uF => messageContentList.Add(MessageContent.FromImageFileId(uF.Id)));

            var threadOptions = new ThreadCreationOptions()
            {
                InitialMessages =
                {
                    new ThreadInitializationMessage(MessageRole.User, messageContentList)
                }
            };

            ThreadRun run = await assistantClient.CreateThreadAndRunAsync(assistant, threadOptions, new RunCreationOptions { Temperature = 0.0f });

            var messages = await GetMessageAsync(run, assistantClient);

            // Debug.Assert(messages.Count == 1);

            var message = messages[0];

            await Console.Out.WriteLineAsync(message.ToString());

            return message;
        }

        #endregion

        private async Task<List<OpenAIFileInfo>> UploadFilesForVisionAsync(IList<string> filePaths)
        {
            var fileClient = _openAIClient.GetFileClient();
            var fileUploadResults = new List<OpenAIFileInfo>();

            foreach (var filePath in filePaths)
            {
                if (File.Exists(filePath))
                {
                    var fileUploadResult = await fileClient.UploadFileAsync(filePath, FileUploadPurpose.Vision);
                    if (fileUploadResult.Value != null)
                    {
                        fileUploadResults.Add(fileUploadResult.Value);
                    }
                }
                else
                {
                    await Console.Out.WriteLineAsync($"Invalid File: {filePath}. Doesn't exist");
                }
            }

            return fileUploadResults;
        }

        private async Task<List<MessageContent>> GetMessageAsync(ThreadRun run, AssistantClient assistantClient)
        {
            while (!run.Status.IsTerminal)
            {
                await Task.Delay(TimeSpan.FromSeconds(1));
                run = await assistantClient.GetRunAsync(run.ThreadId, run.Id);
            }

            if (run.Status == RunStatus.Completed)
            {
                var messagePages = assistantClient.GetMessagesAsync(run.ThreadId);
                var messages = messagePages.AsPages();

                var messageContentList = new List<MessageContent>();
                await foreach (ResultPage<ThreadMessage> resultPage in messages)
                {
                    foreach (var message in resultPage)
                    {
                        messageContentList.AddRange(message.Content);
                    }
                }

                return messageContentList;
            }

            throw new Exception($"GetMessage failed. RunStatus: {run.Status}, {run.LastError}");
        }
    }
}
