namespace Taskforce.Core
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

        public async Task<string> ExecuteAsync(string userPrompt, string content, List<byte[]> images)
        {
            string intermediateResult = content;

            foreach (var agent in _agents)
            {
                if (agent.WithVision)
                {
                    intermediateResult = await agent.ExecuteAsync(userPrompt, intermediateResult, images);
                }
                else
                {
                    intermediateResult = await agent.ExecuteAsync(userPrompt, intermediateResult);
                }

                await Console.Out.WriteColorConsoleAsync(intermediateResult, ConsoleColor.DarkGreen);
            }

            return intermediateResult;
        }
    }
}
