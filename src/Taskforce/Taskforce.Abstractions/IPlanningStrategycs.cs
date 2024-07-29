namespace Taskforce.Abstractions
{
    public interface IPlanningStrategy
    {
        Task<List<string>> PlanAsync(string userPrompt, ILLM llm, string generalInstruction, string answerInstruction);

        Task<List<string>> PlanAsync(string userPrompt, IList<string> imageIds, ILLM llm, string generalInstruction, string answerInstruction);
    }
}
