using System.Text;
using Taskforce.Abstractions;
using Taskforce.Extensions;

namespace Taskforce.Agent
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

        public Agent(LLMBase illm, IMemory shortTermMemory=null, IMemory longTermMemory=null, IPlanning planning=null, IList<ITool> tools=null)
        {
            _llm = illm ?? throw new ArgumentNullException(nameof(illm));
            _shortTermMemory = shortTermMemory ?? throw new ArgumentNullException(nameof(shortTermMemory));
            //_longTermMemory = longTermMemory ?? throw new ArgumentNullException(nameof(longTermMemory));
            _planning = planning ?? throw new ArgumentNullException(nameof(planning));
            //_tools = tools ?? throw new ArgumentNullException(nameof(tools));
        }

        /// <summary>
        /// Describes the role of an agent.
        /// </summary>
        public string Role{ get; set; }

        /// <summary>
        /// The agent's mission. 
        /// </summary>
        public string Mission { get; set; }


        /// <summary>
        /// The agent executes its mission
        /// </summary>
        /// <param name="userPrompt">Can be anything. From a question to an instruction etc.</param>
        /// <param name="content">The content to be processed</param>
        /// <returns>The mission's output</returns>
        public async Task<string> ExecuteAsync(string userPrompt, string content)
        {           
            await Console.Out.WriteAgentLineAsync("Agent starts....");
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
        /// <param name="imagePaths">Files to be uploaded</param>
        /// <returns></returns>
        public async Task<string> ExecuteAsync(string userPrompt, string content, IList<string> imagePaths)
        {
            await Console.Out.WriteAgentLineAsync("Agent starts....");
            var systemPrompt = GetSystemPrompt();
            var instructPrompt = GetInstructPrompt(userPrompt, content);

            // Upload images           
            var imageIds = await _llm.UploadFieAsync(imagePaths);

            // Plan the task.
            var planningResponse = await _planning.PlanAsync(instructPrompt);

            if (planningResponse.Any())
            {
                await ExecuteSubQueriesAsync(content, imageIds, planningResponse);
            }
            
            _shortTermMemory.Store(instructPrompt);

            var context = _shortTermMemory.Get();

            // Get final answer
            var response = await _llm.SendMessageAsync(systemPrompt, context, imageIds);

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

        private async Task ExecuteSubQueriesAsync(string content, IList<string> fileIds, List<string> planningResponse)
        {
            // Execute provided sub tasks/questions from Planner
            var subQuestionAnswer = new StringBuilder();
            if (planningResponse != null && planningResponse.Any())
            {
                foreach (var subquestion in planningResponse)
                {
                    _shortTermMemory.Store($"{subquestion}");
                    await Console.Out.WritePlannerLineAsync("Sub-Question:\n" + subquestion + "\n");

                    var subResponse = await _llm.SendMessageAsync(subquestion, content, fileIds);
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
