namespace Taskforce.Abstractions
{
    /// <summary>
    /// Interacts with a language model to process text
    /// </summary>
    public interface ILLM
    {
        /// <summary>
        /// Sends a prompt to an LLM
        /// </summary>
        /// <param name="systemPrompt"></param>
        /// <param name="userPrompt"></param>
        /// <returns></returns>
        Task<object?> SendMessageAsync(string systemPrompt, string userPrompt);
    }
}