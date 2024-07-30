namespace Taskforce.Abstractions
{
    public interface IReActStrategy
    {
        Task<List<string>> ReasonAndActAsync(string userPrompt, ILLM llm, string generalInstruction, string answerInstruction);

        Task<List<string>> ReasonAndActAsync(string userPrompt, IList<string> imageIds, ILLM llm, string generalInstruction, string answerInstruction);
    }


}
