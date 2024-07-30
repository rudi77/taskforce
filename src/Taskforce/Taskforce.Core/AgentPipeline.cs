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

        public async Task<string> ExecuteAsync(string userPrompt, string content, List<string> imagePaths)
        {
            string intermediateResult = content;

            foreach (var agent in _agents)
            {
                intermediateResult = await agent.ExecuteAsync(userPrompt, intermediateResult, imagePaths);
            }

            return intermediateResult;
        }
    }
}
