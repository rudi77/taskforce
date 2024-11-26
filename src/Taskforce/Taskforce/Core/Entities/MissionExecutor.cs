using Taskforce.Domain.Interfaces;

namespace Taskforce.Core.Entities
{
    public class MissionExecutor
    {
        private readonly IPlanning _planning;
        private readonly LLMBase _llm;
        private readonly MemoryManager _memoryManager;

        public MissionExecutor(IPlanning planning, LLMBase llm, MemoryManager memoryManager)
        {
            _planning = planning;
            _llm = llm;
            _memoryManager = memoryManager;
        }

        public async Task<object?> ExecuteWithPlanningAsync(string instructionPrompt, string content, IList<byte[]> images = null)
        {
            var planningResponse = images == null
                ? await _planning.PlanAsync(instructionPrompt)
                : await _planning.PlanAsync(instructionPrompt, images);

            foreach (var subTask in planningResponse)
            {
                _memoryManager.Store(subTask);
                var subResponse = images == null
                    ? await _llm.SendMessageAsync(subTask, content)
                    : await _llm.SendMessageAsync(subTask, content, images);

                _memoryManager.Store(subResponse.ToString());
            }

            var context = _memoryManager.Retrieve();
            return images == null
                ? await _llm.SendMessageAsync(instructionPrompt, context)
                : await _llm.SendMessageAsync(instructionPrompt, context, images);
        }
    }

}
