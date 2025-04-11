using Taskforce.Domain.Entities;
using Taskforce.Infrastructure.Observability;

namespace Taskforce.Application
{
    public class AgentPipeline
    {
        private readonly List<Agent> _agents;

        public AgentPipeline()
        {
            _agents = new List<Agent>();
        }

        public void AddAgent(Agent agent)
        {
            _agents.Add(agent);
        }

        public async Task<string> ExecuteAsync(string content, List<byte[]> images)
        {
            string intermediateResult = content;

            foreach (var agent in _agents)
            {
                intermediateResult = await agent.ExecuteMissionAsync(agent.Query, intermediateResult, images);
                await Console.Out.WriteColorConsoleAsync(intermediateResult, ConsoleColor.DarkGreen);
            }

            return intermediateResult;
        }
    }
}
