﻿using System.Text;
using Taskforce.Abstractions;

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
        private readonly ILLM _illm;
        private readonly IMemory _shortTermMemory;
        private readonly IMemory _longTermMemory;
        private readonly IPlanning _planning;
        private readonly IList<ITool> _tools;

        public Agent(ILLM illm, IMemory shortTermMemory=null, IMemory longTermMemory=null, IPlanning planning=null, IList<ITool> tools=null)
        {
            _illm = illm ?? throw new ArgumentNullException(nameof(illm));
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
            var systemPrompt = GetSystemPrompt();
            var instructPrompt = GetInstructPrompt(userPrompt, content);

            // Plan
            var planningResponse = await _planning.PlanAsync(instructPrompt);
            await ExecuteSubQueries(content, planningResponse);
            
            _shortTermMemory.Store(instructPrompt);
            var context = _shortTermMemory.Get();

            // Get final answer
            var response = await _illm.SendMessageAsync(systemPrompt, context);

            return response.ToString();
        }

        private async Task ExecuteSubQueries(string content, List<string> planningResponse)
        {
            // Excute provided sub tasks/questions from Planner
            var subQuestionAnswer = new StringBuilder();
            if (planningResponse != null && planningResponse.Any())
            {
                foreach (var subquestion in planningResponse)
                {
                    _shortTermMemory.Store($"{subquestion}");
                    await Console.Out.WriteLineAsync(subquestion + "\n");

                    var subResponse = await _illm.SendMessageAsync(subquestion, content);
                    _shortTermMemory.Store(subResponse.ToString());

                    await Console.Out.WriteLineAsync(subResponse.ToString() + "\n\n");

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
