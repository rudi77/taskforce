using Taskforce.Abstractions;

namespace Taskforce.Core
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
