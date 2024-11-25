using System.Text;
using Taskforce.Domain.Interfaces;
using Taskforce.Infrastructure.Observability;

namespace Taskforce.Domain.Entities
{
    /// <summary>
    /// An Agent manages the main workflow including planning, memory management, LLM
    /// interaction and tool execution.
    /// An Agent has a certain mission, described in a prompt. The agent always tries to
    /// successfully
    /// </summary>
    public class Agent
    {
        private readonly LLMBase _llm;
        private readonly IMemory _shortTermMemory;
        private readonly IMemory _longTermMemory;
        private readonly IPlanning _planning;
        private readonly IList<ITool> _tools;
        private readonly AgentConfig _config;

        public Agent(LLMBase illm, IMemory shortTermMemory, IPlanning planning, AgentConfig config, IMemory longTermMemory = null, IList<ITool> tools = null)
        {
            _llm = illm ?? throw new ArgumentNullException(nameof(illm));
            _shortTermMemory = shortTermMemory ?? throw new ArgumentNullException(nameof(shortTermMemory));
            _planning = planning ?? throw new ArgumentNullException(nameof(planning));
            _config = config ?? throw new ArgumentNullException(nameof(config));

            //_longTermMemory = longTermMemory ?? throw new ArgumentNullException(nameof(longTermMemory));
            //_tools = tools ?? throw new ArgumentNullException(nameof(tools));
        }

        /// <summary>
        /// The Agent's name
        /// </summary>
        public string Name => _config.Name;

        /// <summary>
        /// Describes the role of an agent.
        /// </summary>
        public string Role => _config.Role;

        /// <summary>
        /// The agent's mission. 
        /// </summary>
        public string Mission => _config.Mission;

        /// <summary>
        /// The agent's vision capability
        /// </summary>
        public bool WithVision => _config.WithVision;


        /// <summary>
        /// The agent executes its mission
        /// </summary>
        /// <param name="userPrompt">Can be anything. From a question to an instruction etc.</param>
        /// <param name="content">The content to be processed</param>
        /// <returns>The mission's output</returns>
        public async Task<string> ExecuteAsync(string userPrompt, string content)
        {
            await Console.Out.WriteAgentLineAsync($"Agent: '{Name}' starts....");
            var systemPrompt = GetSystemPrompt();
            var instructPrompt = GetInstructPrompt(userPrompt, content);

            // Plan the task.
            var planningResponse = await _planning.PlanAsync(instructPrompt);

            await ExecuteSubQueriesAsync(content, planningResponse);

            _shortTermMemory.Store(instructPrompt);
            var context = _shortTermMemory.Get();

            // Get final answer
            var response = await _llm.SendMessageAsync(systemPrompt, context);

            return response.ToString();
        }

        /// <summary>
        /// Agent executes its mission with vision support
        /// </summary>
        /// <param name="userPrompt"></param>
        /// <param name="content"></param>
        /// <param name="images">Files to be uploaded</param>
        /// <returns></returns>
        public async Task<string> ExecuteAsync(string userPrompt, string content, IList<byte[]> images)
        {
            await Console.Out.WriteAgentLineAsync($"Agent: '{Name}' starts....");
            var systemPrompt = GetSystemPrompt();
            var instructPrompt = GetInstructPrompt(userPrompt, content);

            // Plan the task.
            var planningResponse = await _planning.PlanAsync(instructPrompt, images);

            if (planningResponse.Any())
            {
                await ExecuteSubQueriesAsync(content, images, planningResponse);
            }

            _shortTermMemory.Store(instructPrompt);

            var context = _shortTermMemory.Get();

            // Get final answer
            var response = await _llm.SendMessageAsync(systemPrompt, context, images);

            return response.ToString();
        }

        private async Task ExecuteSubQueriesAsync(string content, List<string> planningResponse)
        {
            // Execute provided sub tasks/questions from Planner
            var subQuestionAnswer = new StringBuilder();
            if (planningResponse != null && planningResponse.Any())
            {
                foreach (var subquestion in planningResponse)
                {
                    _shortTermMemory.Store($"{subquestion}");

                    await Console.Out.WritePlannerLineAsync("Sub-Question:\n" + subquestion + "\n");

                    var subResponse = await _llm.SendMessageAsync(subquestion, content);

                    _shortTermMemory.Store(subResponse.ToString());

                    await Console.Out.WriteAgentLineAsync("Sub-Response:\n" + subResponse.ToString() + "\n\n");

                    subQuestionAnswer.AppendLine(subquestion).AppendLine(subResponse.ToString());
                }
            }
        }

        private async Task ExecuteSubQueriesAsync(string content, IList<byte[]> images, List<string> planningResponse)
        {
            // Execute provided sub tasks/questions from Planner
            var subQuestionAnswer = new StringBuilder();
            if (planningResponse != null && planningResponse.Any())
            {
                foreach (var subquestion in planningResponse)
                {
                    _shortTermMemory.Store($"{subquestion}");
                    await Console.Out.WritePlannerLineAsync("Sub-Question:\n" + subquestion + "\n");

                    var subResponse = await _llm.SendMessageAsync(Role, GetInstructPrompt(subquestion, content), images);
                    _shortTermMemory.Store(subResponse.ToString());

                    await Console.Out.WriteAgentLineAsync("Sub-Response:\n" + subResponse.ToString() + "\n\n");

                    subQuestionAnswer.AppendLine(subquestion).AppendLine(subResponse.ToString());
                }
            }
        }

        private string GetSystemPrompt()
        {
            return Role + "\n" + Mission;
        }

        private string GetInstructPrompt(string userPrompt, string content)
        {
            return userPrompt + "\n" + content;
        }
    }
}
