using Taskforce.Abstractions;
using Taskforce.Configuration;

namespace Taskforce.Core
{
    public class Planner : IPlanning
    {
        private readonly ILLM _llm;
        private IPlanningStrategy _planningStrategy;
        private readonly PlanningConfig _config;

        public Planner(ILLM llm, IPlanningStrategy planningStrategy, PlanningConfig config)
        {
            _llm = llm ?? throw new ArgumentNullException(nameof(llm));
            _planningStrategy = planningStrategy;
            _config = config;
        }

        public string GeneralInstruction => _config.GeneralInstruction;

        public string AnswerInstruction => _config.AnswerInstruction;

        public void SetStrategy(IPlanningStrategy planningStrategy)
        {
            _planningStrategy = planningStrategy ?? throw new ArgumentNullException(nameof(planningStrategy));
        }

        public async Task<List<string>> PlanAsync(string userPrompt)
        {
            await Console.Out.WritePlannerLineAsync("Planner starts planning...");
            return await _planningStrategy.PlanAsync(userPrompt, _llm, GeneralInstruction, AnswerInstruction);
        }

        public async Task<List<string>> PlanAsync(string userPrompt, IList<string> imagePaths)
        {
            await Console.Out.WritePlannerLineAsync("Planner starts planning...");
            return await _planningStrategy.PlanAsync(userPrompt, imagePaths, _llm, GeneralInstruction, AnswerInstruction);
        }
    }
}
