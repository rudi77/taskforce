//using OpenAI;
//using OpenAI.Chat;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Taskforce.Domain.Interfaces;

namespace Taskforce.Infrastructure.LLM
{
    public class OpenAIChatClient : LLMBase
    {
        //private readonly ChatClient _chatClient;
        private readonly Kernel _kernel;

        public OpenAIChatClient(string apiKey)
        {
            _kernel = BuildKernel("gpt-4o-mini", apiKey);
        }

        public OpenAIChatClient()
        {
            // read OpenAIApiKey from environment variable
            var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");

            if (string.IsNullOrEmpty(apiKey))
            {
                throw new InvalidOperationException("OPENAI_API_KEY not set as environment variable");
            }

            _kernel = BuildKernel("gpt-4o-mini", apiKey);
        }

        private Kernel BuildKernel(string modelId, string apiKey)
        {
            var kernelBuilder = Kernel.CreateBuilder();
            kernelBuilder.AddOpenAIChatCompletion(
                modelId: modelId,
                apiKey: apiKey
            );

            var kernel = kernelBuilder.Build();

            return kernel;
        }


        public override async Task<object?> SendMessageAsync(string systemPrompt, string userPrompt)
        {
            ChatHistory chatHistory = [];
            chatHistory.Add(
                new()
                {
                    Role = AuthorRole.System,
                    Items = [new TextContent { Text = systemPrompt }]
                }
            );

            chatHistory.Add(
                new()
                {
                    Role = AuthorRole.User,
                    Items = [new TextContent { Text = userPrompt }]
                }
            );

            var chatCompletionService = _kernel.GetRequiredService<IChatCompletionService>();

            ChatMessageContent? chatCompletion = await chatCompletionService.GetChatMessageContentAsync(chatHistory);

            return chatCompletion.Items[0].ToString();
        }

        public override async Task<object?> SendMessageAsync(string systemPrompt, string userPrompt, IList<byte[]> images)
        {
            ChatHistory chatHistory = [];
            chatHistory.Add(
                new()
                {
                    Role = AuthorRole.System,
                    Items = [
                        new TextContent { Text = systemPrompt },
                    ]
                }
            );

            chatHistory.Add(
                new()
                {
                    Role = AuthorRole.User,
                    Items = [
                        new TextContent { Text = userPrompt },
                        new ImageContent(images[0], "image/png")
                    ]
                }
            );

            var chatCompletionService = _kernel.GetRequiredService<IChatCompletionService>();

            ChatMessageContent? chatCompletion = await chatCompletionService.GetChatMessageContentAsync(chatHistory);

            return chatCompletion.Items[0].ToString();
        }
    }
}
