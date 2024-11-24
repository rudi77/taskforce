namespace Taskforce.Abstractions
{
    public interface IPlanningStrategy
    {
        Task<List<string>> PlanAsync(string userPrompt, IChatCompletion llm, string generalInstruction, string answerInstruction);

        Task<List<string>> PlanAsync(string userPrompt, IList<byte[]> images, IChatCompletion llm, string generalInstruction, string answerInstruction);
    }
}
