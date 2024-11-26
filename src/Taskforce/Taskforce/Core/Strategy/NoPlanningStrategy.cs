using Taskforce.Domain.Interfaces;

namespace Taskforce.Domain.Services
{
    public class NoPlanningStrategy : IPlanningStrategy
    {
        public async Task<List<string>> PlanAsync(string userPrompt, IChatCompletion llm, string generalInstruction, string answerInstruction)
        {
            return new List<string>();
        }

        public async Task<List<string>> PlanAsync(string userPrompt, IList<byte[]> images, IChatCompletion llm, string generalInstruction, string answerInstruction)
        {
            return new List<string>();
        }
    }
}
