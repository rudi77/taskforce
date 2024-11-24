namespace Taskforce.Abstractions
{
    public interface IReActStrategy
    {
        Task<List<string>> ReasonAndActAsync(string userPrompt, IChatCompletion llm, string generalInstruction, string answerInstruction);

        Task<List<string>> ReasonAndActAsync(string userPrompt, IList<string> imageIds, IChatCompletion llm, string generalInstruction, string answerInstruction);
    }


}
