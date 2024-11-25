using Taskforce.Domain.Interfaces;

namespace Taskforce.Domain.Services
{
    public class NoPlanPlanner : IPlanning
    {
        public async Task<List<string>> PlanAsync(string userPrompt)
        {
            return new List<string>();
        }

        public async Task<List<string>> PlanAsync(string userPrompt, IList<byte[]> images)
        {
            return new List<string>();
        }
    }
}
