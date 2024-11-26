using Microsoft.Extensions.Logging;
using System.Buffers;
using System.Text;
using Taskforce.Core.Entities;
using Taskforce.Domain.Interfaces;
using Taskforce.Infrastructure.Observability;

namespace Taskforce.Domain.Entities
{
    public class Agent
    {
        private readonly LLMBase _llm;
        private readonly IPlanning _planning;
        private readonly MemoryManager _memoryManager;
        private readonly ILogger _logger;
        private readonly PromptBuilder _promptBuilder;

        public string Name { get; }
        public string Role { get; }
        public string Mission { get; }
        public bool WithVision { get; }

        public Agent(LLMBase llm, IPlanning planning, AgentConfig config, MemoryManager memoryManager, PromptBuilder promptBuilder, ILogger logger)
        {
            _llm = llm ?? throw new ArgumentNullException(nameof(llm));
            _planning = planning ?? throw new ArgumentNullException(nameof(planning));
            _memoryManager = memoryManager ?? throw new ArgumentNullException(nameof(memoryManager));
            _promptBuilder = promptBuilder ?? throw new ArgumentNullException(nameof(promptBuilder));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));

            Name = config.Name;
            Role = config.Role;
            Mission = config.Mission;
            WithVision = config.WithVision;
        }

        /// <summary>
        /// Executes the mission by planning and executing steps.
        /// </summary>
        public async Task<string> ExecuteMissionAsync(string userPrompt, string content, IList<byte[]> images = null)
        {
            _logger.LogInformation($"Agent '{Name}' is starting its mission: {Mission}");

            // Plan the mission steps
            var plan = await PlanMissionAsync(userPrompt, content, images);

            // Execute each step and aggregate results
            var finalResult = await ExecutePlanyAsync(plan, content, images);

            _logger.LogInformation($"Agent '{Name}' completed its mission.");

            return finalResult;
        }

        private async Task<List<string>> PlanMissionAsync(string userPrompt, string content, IList<byte[]> images)
        {
            try
            {
                return WithVision
                    ? await _planning.PlanAsync(userPrompt, images)
                    : await _planning.PlanAsync(userPrompt);
                    
            }
            catch (Exception ex)
            {
                _logger.LogError($"Error while planning mission: {ex.Message}", ex);
                throw;
            }
        }

        private async Task<string> ExecutePlanyAsync(List<string> plan, string content, IList<byte[]> images)
        {
            var results = new StringBuilder();
            var systemPrompt = _promptBuilder.BuildSystemPrompt();

            foreach (var step in plan)
            {
                _memoryManager.Store(step);
                _logger.LogDebug($"Executing step: {step}");
                var instructPrompt = _promptBuilder.BuildInstructionPrompt(step, content);

                var response = WithVision
                    ? await _llm.SendMessageAsync(systemPrompt, instructPrompt, images)
                    : await _llm.SendMessageAsync(systemPrompt, instructPrompt);
                    

                _memoryManager.Store(response.ToString());
                results.AppendLine(response.ToString());

                _logger.LogDebug($"Step result: {response}");
            }

            return results.ToString();
        }
    }

}
