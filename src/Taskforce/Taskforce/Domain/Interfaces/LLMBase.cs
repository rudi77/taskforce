namespace Taskforce.Domain.Interfaces
{
    public abstract class LLMBase : IChatCompletion
    {
        public abstract Task<object?> SendMessageAsync(string systemPrompt, string userPrompt);

        public abstract Task<object?> SendMessageAsync(string systemPrompt, string userPrompt, IList<byte[]> images);
    }
}
