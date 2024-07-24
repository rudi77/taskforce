using OpenAI.Chat;
using OpenAI.Models;
using OpenAI;

using Taskforce.Abstractions;

namespace Taskforce.LLM
{
    public class OpenAILLM : ILLM
    {
        private readonly OpenAIClient _client;

        public OpenAILLM(string apiKey)
        {
            _client = new OpenAIClient(apiKey);
        }

        public OpenAILLM()
        {
            // read OpenAIApiKey from environment variable
            var apiKey = Environment.GetEnvironmentVariable("OpenAIApiKey");

            if (string.IsNullOrEmpty(apiKey))
            {
                throw new InvalidOperationException("OpenAIApiKey not set as environment variable");
            }

            _client = new OpenAIClient(apiKey);
        }

        public async Task<object?> SendMessageAsync(string systemPrompt, string userPrompt)
        {
            var messages = CreateMessages(systemPrompt, userPrompt);

            // TODO: make the model configurable
            var chatRequest = new ChatRequest(messages, Model.GPT4o /*.GPT3_5_Turbo*/, temperature: 0.0);
            var response = await _client.ChatEndpoint.GetCompletionAsync(chatRequest);

            return response.FirstChoice.ToString();
        }

        private static List<Message> CreateMessages(string systemPrompt, string userPrompt)
        {
            //var userPrompt = instructPrompt.Replace("{Content}", content);

            return new List<Message>
            {
                new(Role.System, systemPrompt),
                new(Role.User, userPrompt)
            };
        }
    }
}
