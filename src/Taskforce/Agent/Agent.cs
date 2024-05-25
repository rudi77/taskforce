﻿using Taskforce.Abstractions;

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

        public Agent(ILLM illm, IMemory shortTermMemory, IMemory longTermMemory, IPlanning planning, IList<ITool> tools)
        {
            _illm = illm ?? throw new ArgumentNullException(nameof(illm));
            _shortTermMemory = shortTermMemory ?? throw new ArgumentNullException(nameof(shortTermMemory));
            _longTermMemory = longTermMemory ?? throw new ArgumentNullException(nameof(longTermMemory));
            _planning = planning ?? throw new ArgumentNullException(nameof(planning));
            _tools = tools ?? throw new ArgumentNullException(nameof(tools));
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
        /// <param name="userPrompt">Any text based content that shall be processed by this agent</param>
        /// <returns>The mission's output</returns>
        public async Task<string> ExecuteAsync(string userPrompt)
        {
            var systemPrompt = GetSystemPrompt();

            var response = await _illm.SendMessageAsync(systemPrompt, userPrompt);

            return response.ToString();
        }

        private string GetSystemPrompt()
        {
            return Role + "\n" + Mission;
        }
    }
}
