using Taskforce.Abstractions;

namespace Planning
{
    public class NoPlanPlanner : IPlanning
    {
        public async Task<List<string>> PlanAsync(string userPrompt)
        {
            return new List<string>();
        }

        public async Task<List<string>> PlanAsync(string userPrompt, IList<string> imagePaths)
        {
            return new List<string>();
        }
    }
}
