namespace Taskforce.Abstractions
{
    public abstract class LLMBase : ILLM, ILLMImageUpload
    {
        public abstract Task<object?> SendMessageAsync(string systemPrompt, string userPrompt);

        public abstract Task<object?> SendMessageAsync(string systemPrompt, string userPrompt, IList<string> fileIds);

        public abstract Task<string[]> UploadFieAsync(IList<string> filePaths);
    }
}
