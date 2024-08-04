using Taskforce.Abstractions;
using OpenAI.Assistants;
using System.ClientModel;
using OpenAI.Files;
using OpenAI;
using System.Diagnostics;

namespace Taskforce.Core
{
    public class OpenAIAssistantClient : LLMBase
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

        public override async Task<object?> SendMessageAsync(string systemPrompt, string userPrompt)
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
            var messages = await GetMessageAsync(run, assistantClient);

            // Debug.Assert(messages.Count == 1);

            var message = messages[0];

            await Console.Out.WriteLineAsync(message.ToString());

            return message;
        }

        public override async Task<object?> SendMessageAsync(string systemPrompt, string userPrompt, IList<string> imageIds)
        {
            var assistantClient = _openAIClient.GetAssistantClient();
            //var assistant = await assistantClient.CreateAssistantAsync("gpt-3.5-turbo", new AssistantCreationOptions { Instructions = systemPrompt });
            var assistant = await assistantClient.CreateAssistantAsync("gpt-4o-mini", new AssistantCreationOptions { Instructions = systemPrompt });
            var messageContentList = string.IsNullOrEmpty(userPrompt) 
                ? new List<MessageContent>() 
                : new List<MessageContent> { userPrompt };

            foreach (var fileId in imageIds)
            {
                messageContentList.Add(MessageContent.FromImageFileId(fileId));
            }


            var threadOptions = new ThreadCreationOptions()
            {
                InitialMessages =
                {
                    new ThreadInitializationMessage(MessageRole.User, messageContentList)
                },               
            };



            ThreadRun run = await assistantClient.CreateThreadAndRunAsync(assistant, threadOptions, new RunCreationOptions { Temperature = 0.0f });

            var messages = await GetMessageAsync(run, assistantClient);

            // Debug.Assert(messages.Count == 1);

            var message = messages[0];

            await Console.Out.WriteLineAsync(message.ToString());

            return message;
        }

        public override async Task<string[]> UploadFieAsync(IList<string> imagePaths)
        {
            var uploadedFiles = await UploadFilesForVisionAsync(imagePaths);

            return uploadedFiles.Select(fi => fi.Id).ToArray();
        }

        #endregion

        private async Task<List<OpenAIFileInfo>> UploadFilesForVisionAsync(IList<string> imagePaths)
        {
            var fileClient = _openAIClient.GetFileClient();
            var fileUploadResults = new List<OpenAIFileInfo>();

            foreach (var filePath in imagePaths)
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
