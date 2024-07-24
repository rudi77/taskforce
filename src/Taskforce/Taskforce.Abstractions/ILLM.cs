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

        /// <summary>
        /// Sends a prompt with images includes to an MMLM
        /// </summary>
        /// <param name="systemPrompt"></param>
        /// <param name="userPrompt"></param>
        /// <param name="images"></param>
        /// <returns></returns>
        Task<object?> SendMessageAsync(string systemPrompt, string userPrompt, IList<byte[]> images);
    }
}