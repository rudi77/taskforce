using Taskforce.Domain.Entities;

namespace Taskforce.Core.Entities
{
    public class PromptBuilder
    {
        private readonly AgentConfig _config;

        public PromptBuilder(AgentConfig config)
        {
            _config = config;
        }

        public string BuildSystemPrompt()
        {
            return $"{_config.Role}\n{_config.Mission}";
        }

        public string BuildInstructionPrompt(string userPrompt, string content)
        {
            return $"{userPrompt}\n{content}";
        }
    }

}
