using Taskforce.Abstractions;
using OpenAI.Assistants;
using System.ClientModel;

namespace Taskforce.LLM
{
    public class OpenAIAssistantClient : ILLM
    {
#pragma warning disable OPENAI001
                private readonly AssistantClient _assistantClient;

        public OpenAIAssistantClient(string apiKey)
        {
            _assistantClient = new AssistantClient(apiKey);
        }

        public OpenAIAssistantClient()
        {
            // read OpenAIApiKey from environment variable
            var apiKey = Environment.GetEnvironmentVariable("OpenAIApiKey");

            if (string.IsNullOrEmpty(apiKey))
            {
                throw new InvalidOperationException("OpenAIApiKey not set as environment variable");
            }

            _assistantClient = new AssistantClient(apiKey);
        }

        #region ILLM interface methods

        public async Task<object?> SendMessageAsync(string systemPrompt, string userPrompt)
        {
            var assistant = await _assistantClient.CreateAssistantAsync(
                "gpt-4o-mini",
                new AssistantCreationOptions()
                {
                    Instructions = systemPrompt,
                });

            var threadOptions = new ThreadCreationOptions
            {
                InitialMessages = { userPrompt }
            };

            ThreadRun run = await _assistantClient.CreateThreadAndRunAsync(assistant, threadOptions, new RunCreationOptions { Temperature = 0.0f });

            
            while (!run.Status.IsTerminal)
            {
                await Task.Delay(TimeSpan.FromSeconds(1));
                run = await _assistantClient.GetRunAsync(run.ThreadId, run.Id);
            }

            if (run.Status == RunStatus.Completed)
            {
                var messagePages = _assistantClient.GetMessagesAsync(run.ThreadId);

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

        public Task<object?> SendMessageAsync(string systemPrompt, string userPrompt, IList<byte[]> images)
        {
            throw new NotImplementedException();
        }

        #endregion
    }
}
