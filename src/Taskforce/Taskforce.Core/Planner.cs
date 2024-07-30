using Taskforce.Abstractions;

namespace Taskforce.Core
{
    public class Planner : IPlanning
    {
        private readonly ILLM _llm;
        private IPlanningStrategy _planningStrategy;

        public Planner(ILLM llm, IPlanningStrategy planningStrategy)
        {
            _llm = llm ?? throw new ArgumentNullException(nameof(llm));
            _planningStrategy = planningStrategy;
        }

        public string GeneralInstruction { get; set; }

        public string AnswerInstruction { get; set; }

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
